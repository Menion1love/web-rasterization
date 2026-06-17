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

struct SingleGaussianGrad {
  opacity: atomic<i32>, 
  center_x: atomic<i32>,
  center_y: atomic<i32>,
  cov_a: atomic<i32>,  
  cov_b: atomic<i32>,  
  cov_c: atomic<i32>,  
  color_r: atomic<i32>,
  color_g: atomic<i32>,
  color_b: atomic<i32>,
  
  pad0: i32,
  pad1: i32,
  pad2: i32,
}

struct GaussianGradients {
  grads: array<SingleGaussianGrad>, 
}

@group(0) @binding(0) var<uniform> camera : camData;
@group(0) @binding(1) var<storage, read> inputData: array<rasterData>;
@group(0) @binding(2) var<storage, read_write> valuesBuffer: array<u32>;
@group(0) @binding(3) var<storage, read_write> tail_ranges: array<tailData>;
@group(0) @binding(4) var tex_gradients: texture_2d<f32>;
@group(0) @binding(5) var tex_transsmitance: texture_2d<f32>;
@group(0) @binding(6) var<storage, read_write> global_grads: GaussianGradients;

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

  var transmittance = textureLoad(tex_transsmitance, pixel_coord, 0).r;
  let dL_dI = textureLoad(tex_gradients, pixel_coord, 0).rgb;

  var C_after = vec3f(0.0);

  var loop_counter = 0;

  for (var i = i32(end_idx); i >= i32(start_idx); i -= 1) {
    let g_id = valuesBuffer[i];

    let data = inputData[g_id];
    
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
    
    if (power > 0.0) { 
      continue; 
    } 
    
    let alpha = data.opacity * exp(power);
    if (alpha < 0.005) { 
      continue;
    }
    
    let denom = max(1.0 - alpha, 0.0001);
    transmittance = transmittance / denom;

    let dL_dcolor = dL_dI * alpha * transmittance;

    let dL_dalpha_vector = dL_dI * (data.color * transmittance - C_after);
    let dL_dalpha: f32 = dL_dalpha_vector.r + dL_dalpha_vector.g + dL_dalpha_vector.b;

    let dL_dopacity = dL_dalpha * exp(power);
    let dL_dpower   = dL_dalpha * alpha;

    let dp_dc_x = d.x * inv_cov_00 + d.y * inv_cov_01;
    let dp_dc_y = d.x * inv_cov_01 + d.y * inv_cov_11;
    let dL_dcenter_2d = vec2f(dp_dc_x, dp_dc_y) * (-dL_dpower);

    let dL_dinv_cov_00 = -0.5 * d.x * d.x * dL_dpower;
    let dL_dinv_cov_01 = -1.0 * d.x * d.y * dL_dpower;
    let dL_dinv_cov_11 = -0.5 * d.y * d.y * dL_dpower;

    let dL_dinv = mat2x2f(
        -dL_dinv_cov_00, -dL_dinv_cov_01,
        -dL_dinv_cov_01, -dL_dinv_cov_11
    );
    
    let inv_cov = mat2x2f(inv_cov_00, inv_cov_01, inv_cov_01, inv_cov_11);
    
    let dL_dcov = inv_cov * dL_dinv * inv_cov;

    let dL_da = dL_dcov[0][0];
    let dL_db = dL_dcov[0][1] + dL_dcov[1][0]; 
    let dL_dc = dL_dcov[1][1];
    let dL_dcov_2d = vec3f(dL_da, dL_db, dL_dc);

    let scale = 1000000.0; 

    atomicAdd(&global_grads.grads[g_id].color_r, i32(round(0 * dL_dcolor.r * scale)));
    atomicAdd(&global_grads.grads[g_id].color_g, i32(round(0 * dL_dcolor.g * scale)));
    atomicAdd(&global_grads.grads[g_id].color_b, i32(round(0 * dL_dcolor.b * scale)));

    atomicAdd(&global_grads.grads[g_id].opacity, i32(round(0 * dL_dopacity * scale)));

    atomicAdd(&global_grads.grads[g_id].center_x, i32(round(0 * dL_dcenter_2d.x * scale)));
    atomicAdd(&global_grads.grads[g_id].center_y, i32(round(0 * dL_dcenter_2d.y * scale)));

    atomicAdd(&global_grads.grads[g_id].cov_a, i32(round(0 * dL_dcov_2d.x * scale)));
    atomicAdd(&global_grads.grads[g_id].cov_b, i32(round(0 * dL_dcov_2d.y * scale)));
    atomicAdd(&global_grads.grads[g_id].cov_c, i32(round(0 * dL_dcov_2d.z * scale)));

    C_after = C_after * (1.0 - alpha) + data.color * alpha * transmittance;

    if (transmittance < 0.0001) { break; }
    loop_counter++;
    // if (loop_counter > 100000) { break; }
  }
}
