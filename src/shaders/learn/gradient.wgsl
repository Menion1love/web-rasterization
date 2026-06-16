@group(0) @binding(0) var tex_render: texture_2d<f32>;
@group(0) @binding(1) var tex_ground_truth: texture_2d<f32>;
@group(0) @binding(2) var tex_final_gradients: texture_storage_2d<rgba32float, write>;

var<workgroup> shared_render: array<array<vec3f, 26>, 26>;
var<workgroup> shared_truth:  array<array<vec3f, 26>, 26>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(local_invocation_id) local_id: vec3u,
  @builtin(workgroup_id) workgroup_id: vec3u,
  @builtin(global_invocation_id) global_id: vec3u
) {
  let texture_size = vec2i(textureDimensions(tex_render));
  let tile_start = vec2i(workgroup_id.xy * 16u) - vec2i(5, 5);

  for (var y = i32(local_id.y); y < 26; y += 16) {
    for (var x = i32(local_id.x); x < 26; x += 16) {
      let global_coord = tile_start + vec2i(x, y);
      let clamped_coord = clamp(global_coord, vec2i(0, 0), texture_size - vec2i(1, 1));
      
      shared_render[y][x] = textureLoad(tex_render, clamped_coord, 0).rgb;
      shared_truth[y][x]  = textureLoad(tex_ground_truth, clamped_coord, 0).rgb;
    }
  }
  workgroupBarrier(); 
  if (global_id.x >= u32(texture_size.x) || global_id.y >= u32(texture_size.y)) { 
    return; 
  }

  let sh_x = i32(local_id.x) + 5;
  let sh_y = i32(local_id.y) + 5;

  var sum_x  = vec3f(0.0); var sum_y  = vec3f(0.0);
  var sum_x2 = vec3f(0.0); var sum_y2 = vec3f(0.0);
  var sum_xy = vec3f(0.0);

  for (var dy = -5; dy <= 5; dy++) {
    for (var dx = -5; dx <= 5; dx++) {
      let r_color = shared_render[sh_y + dy][sh_x + dx];
      let t_color = shared_truth[sh_y + dy][sh_x + dx];
      
      sum_x  += r_color;
      sum_y  += t_color;
      sum_x2 += r_color * r_color;
      sum_y2 += t_color * t_color; 
      sum_xy += r_color * t_color;
    }
  }

  let w = 1.0 / 121.0;
  let C1 = 0.0001;
  let C2 = 0.0009;

  let I_x = shared_render[sh_y][sh_x];
  let I_y = shared_truth[sh_y][sh_x];

  let mx = sum_x.r * w;
  let my = sum_y.r * w;
  let sx_sq = (sum_x2.r * w) - (mx * mx);
  let sy_sq = (sum_y2.r * w) - (my * my);
  let sxy   = (sum_xy.r * w) - (mx * my);

  let A = 2.0 * mx * my + C1;
  let B = 2.0 * sxy + C2;
  let D = mx * mx + my * my + C1;
  let E = sx_sq + sy_sq + C2; 

  let d_num = 2.0 * my * B + 2.0 * (I_y.r - mx) * A;
  let d_den = 2.0 * mx * E + 2.0 * (I_x.r - mx) * D;

  let dSSIM_dIx = w * ((d_num * E - A * B * d_den / D) / (D * E * E)); 
    
  var dL_dI_dssim = vec3f(0.0);
  dL_dI_dssim.r = -0.5 * dSSIM_dIx;
  dL_dI_dssim.g = dL_dI_dssim.r; 
  dL_dI_dssim.b = dL_dI_dssim.r;

  let dL_dI_l1 = (I_x - I_y);

  let dL_dI_final = 0.8 * dL_dI_l1 + 0.2 * dL_dI_dssim;

  textureStore(tex_final_gradients, vec2i(global_id.xy), vec4f(dL_dI_final, 1.0));
}
