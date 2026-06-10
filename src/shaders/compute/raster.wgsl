struct camData {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  vp: mat4x4<f32>,
  locW: vec4<f32>,
  atH: vec4<f32>,
  dirProjDist: vec4<f32>,
  rightWp: vec4<f32>,
  upHp: vec4<f32>,
}

struct rasterData {
  center_2d: vec2<f32>,
  color: vec3<f32>,
  opacity: f32,
  cov_2d: vec3<f32>,
  depth: f32,
  index: u32,
  dummy: f32,
}

struct sortKey {
  key: u32,
  index: u32,
  padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> camera : camData;
@group(0) @binding(1) var<storage, read> inputData: array<rasterData>;
@group(0) @binding(2) var<storage, read_write> keysBuffer: array<sortKey>;
@group(0) @binding(3) var tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>, @builtin(global_invocation_id) global_id : vec3<u32>) {
  let pixel_coord = global_id.xy;
  let tile_size = 16.0;
  let grid_width = camera.locW.w / tile_size;
  let grid_height = camera.atH.w / tile_size;

  let tile_id = f32(workgroup_id.y) * grid_width + f32(workgroup_id.x);
  
  // let range = tile_ranges[tile_id]; 
  let start_idx = 0;
  let end_idx = 100;

  var final_color = vec3<f32>(0.0, 0.0, 0.0);
  var transmittance = 1.0;

  for (var i = start_idx; i < end_idx; i++) {
    let ga_index = keysBuffer[i].index;
    
    let data = inputData[ga_index];
    
    let d = vec2<f32>(pixel_coord) - data.center_2d;
    
    let a = data.cov_2d.x;
    let b = data.cov_2d.y;
    let c = data.cov_2d.z;
    
    let det = a * c - b * b;
    if (det <= 0.0) { continue; }

    let inv_det = 1.0 / det;
    let inv_cov_00 = c * inv_det;
    let inv_cov_01 = -b * inv_det;
    let inv_cov_11 = a * inv_det;
    
    let power = -0.5 * (d.x * d.x * inv_cov_00 + 2.0 * d.x * d.y * inv_cov_01 + d.y * d.y * inv_cov_11);
    
    if (power > 0.0) { continue; } 
    
    let alpha = data.opacity * exp(power);
    
    if (alpha < 1.0 / 255.0) { continue; }
    
    let weight = alpha * transmittance;
    final_color += data.color * weight;
    
    transmittance *= (1.0 - alpha);
    
    if (transmittance < 0.0001) {
        break;
    }
  }
  textureStore(tex, pixel_coord, vec4<f32>(final_color, 1.0));
}
