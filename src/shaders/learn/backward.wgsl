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
  padding: vec2<f32>,
  color: vec3<f32>,
  opacity: f32,
  cov_2d: vec3<f32>,
  depth: u32,
  min_tile: vec2<i32>,
  max_tile: vec2<i32>,
}

struct tailData {
  start_idx: u32,
  end_idx: u32,
}

@group(0) @binding(0) var<uniform> camera : camData;
@group(0) @binding(1) var<storage, read> inputData: array<rasterData>;
@group(0) @binding(2) var<storage, read_write> valuesBuffer: array<u32>;
@group(0) @binding(3) var<storage, read_write> tail_ranges: array<tailData>;
@group(0) @binding(4) var tex_gradients: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var tex_transsmitance: texture_storage_2d<rgba32float, read_write>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(local_invocation_id) local_id: vec3u,
  @builtin(workgroup_id) workgroup_id: vec3u,
  @builtin(global_invocation_id) global_id: vec3u
) {
  let pixel_coord = global_id.xy;
  let tile_size = 16u;
  let grid_width =  (u32(camera.locW.w) / tile_size); 
  let grid_height = (u32(camera.atH.w) / tile_size);

  let tile_x = u32(global_id.x / tile_size);
  let tile_y = u32(global_id.y / tile_size);

  let tile_id = tile_y * u32(grid_width) + tile_x;

  let range = tail_ranges[tile_id]; 
  let start_idx = range.start_idx;
  let end_idx = range.end_idx;

  let transmittance = textureLoad(tex_transsmitance, pixel_coord, 0).r;
  let dL_dI = textureLoad(tex_gradients, pixel_coord, 0).rgb;

  let col = vec3f(0.0);

  for (var i = end_idx; i >= start_idx; i--) {
    let data = inputData[valuesBuffer[i]];
    
    let d = vec2<f32>(pixel_coord) - data.center_2d;
    
    let a = data.cov_2d.x;
    let b = data.cov_2d.y;
    let c = data.cov_2d.z;
    
    let det = a * c - b * b;

    let inv_det = 1.0 / det;
    let inv_cov_00 = c * inv_det;
    let inv_cov_01 = -b * inv_det;
    let inv_cov_11 = a * inv_det;
    
    let power = -0.5 * (d.x * d.x * inv_cov_00 + 2.0 * d.x * d.y * inv_cov_01 + d.y * d.y * inv_cov_11);
    let alpha = data.opacity * exp(power);
    
    let denom = max(1.0 - alpha, 0.0001);
    let dL_dalpha = dL_dalpha_vector.r + dL_dalpha_vector.g + dL_dalpha_vector.b;

    col = data.color * alpha * transmittance;

    let dL_dopacity = dL_dalpha * exp(power);
    let dL_dpower   = dL_dalpha * alpha; 

    let dpower_dcenter = vec2f(0.0);
    dpower_dcenter.x = (d.x * inv_cov_00 + d.y * inv_cov_01);
    dpower_dcenter.y = (d.x * inv_cov_01 + d.y * inv_cov_11);

    let dL_dcenter_2d = dL_dpower * dpower_dcenter; 

    let dL_dinv_cov_00 = -0.5 * dL_dpower * d.x * d.x;
    let dL_dinv_cov_01 = -0.5 * dL_dpower * 2.0 * d.x * d.y; 
    let dL_dinv_cov_11 = -0.5 * dL_dpower * d.y * d.y;

    let dL_da = inv_det * (inv_cov_01 * dL_dinv_cov_01 + inv_cov_11 * dL_dinv_cov_11) - inv_cov_00 * (inv_cov_00 * dL_dinv_cov_00 + inv_cov_01 * dL_dinv_cov_01 + inv_cov_11 * dL_dinv_cov_11);
    let dL_db = 2.0 * inv_det * (inv_cov_00 * dL_dinv_cov_01 + inv_cov_01 * dL_dinv_cov_11) - 2.0 * inv_cov_01 * (inv_cov_00 * dL_dinv_cov_00 + inv_cov_01 * dL_dinv_cov_01 + inv_cov_11 * dL_dinv_cov_11);
    let dL_dc = inv_det * (inv_cov_00 * dL_dinv_cov_00 + inv_cov_01 * dL_dinv_cov_01) - inv_cov_11 * (inv_cov_00 * dL_dinv_cov_00 + inv_cov_01 * dL_dinv_cov_01 + inv_cov_11 * dL_dinv_cov_11);

    let dL_dcov_2d = vec3f(dL_da, dL_db, dL_dc);

    // transmittance = transmittance / (1.0 - alpha);

    // let dL_dc = dL_dI * transmittance * alpha;
    // let dL_da = dL_dI * (data.color * transmittance - col / (1.0 - alhpa));


    
    let data = inputData[valuesBuffer[i]];
  }
}
