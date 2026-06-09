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

struct gaussainData {
  position: vec3<f32>,
  scale: f32,
  rotation: vec4<f32>,
  color: vec4<f32>,
  opacity: f32,
  index: u32,
  padding: vec2<f32>,
}

struct rasterData {
  center_2d: vec2<f32>,
  color: vec3<f32>,
  opacity: f32,
  cov_2d: vec3<f32>,
  depth: f32,
  index: u32,
  padding: f32,
}

struct sortKey {
  key: u32,
  index: u32,
  padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera : camData;
@group(0) @binding(1) var<storage, read> inputData: array<gaussainData>;
@group(0) @binding(2) var<storage, read_write> outputData: array<rasterData>;
@group(0) @binding(3) var<storage, read_write> keysBuffer: array<sortKey>;
@group(0) @binding(4) var<storage, read_write> global_keys_counter: atomic<u32>;

fn quaternion_to_matrix(q_input: vec4<f32>) -> mat3x3<f32> {
  let q = normalize(q_input);
  
  let x = q.x;
  let y = q.y;
  let z = q.z;
  let w = q.w;

  let xx = x * x;
  let yy = y * y;
  let zz = z * z;
  
  let xy = x * y;
  let xz = x * z;
  let yz = y * z;
  
  let wx = w * x;
  let wy = w * y;
  let wz = w * z;

  return mat3x3<f32>(
      vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz),       2.0 * (xz - wy)),
      vec3<f32>(2.0 * (xy - wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
      vec3<f32>(2.0 * (xz + wy),       2.0 * (yz - wx),       1.0 - 2.0 * (xx + yy))
  );
}

fn compute_covariance(s: vec3<f32>, r: vec4<f32>) -> mat3x3<f32> {
  let R = quaternion_to_matrix(r);

  var M: mat3x3<f32>;
  M[0] = R[0] * s.x; 
  M[1] = R[1] * s.y; 
  M[2] = R[2] * s.z; 

  return M * transpose(M);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;

  if (idx >= arrayLength(&inputData)) {
    return; 
  }

  let gs = inputData[idx];
  let t = camera.view * vec4<f32>(gs.position, 1.0);
  var out: rasterData;

  if (t.z < 0.2) {
    return;
  }

  out.depth = t.z;
  let half_w = f32(camera.locW.w) * 0.5;
  let half_h = f32(camera.atH.w) * 0.5;

  let fx = camera.proj[0][0] * half_w;
  let fy = camera.proj[1][1] * half_h;

  out.center_2d.x = (t.x / t.z) * fx + half_w; 
  out.center_2d.y = -(t.y / t.z) * fy + half_h;

  let tz2 = t.z * t.z;

  let cov_3d = compute_covariance(vec3<f32>(gs.scale), gs.rotation);

  let J = mat3x3<f32>(
    vec3<f32>(fx / t.z, 0, -(t.x * fx) / tz2),
    vec3<f32>(0, fy / t.z, -(t.y * fy) / tz2),
    vec3<f32>(0, 0, 1),
  );

  let w = mat3x3<f32>(
    camera.view[0].xyz,
    camera.view[1].xyz,
    camera.view[2].xyz
  );

  let cov_2d = J * w * cov_3d * transpose(w) * transpose(J);

  let fil = 0.3; 
  out.cov_2d = vec3<f32>(cov_2d[0][1] + fil, cov_2d[0][0], cov_2d[1][0] + fil);
  out.opacity = gs.opacity;
  out.index = gs.index;
  out.color = gs.color.xyz;

  let a = out.cov_2d.x;
  let b = out.cov_2d.y;
  let c = out.cov_2d.z;

  let trace = a + c;
  let det = a * c - b * b;

  let mid = 0.5 * trace;
  let term = sqrt(max(0.1, mid * mid - det));
  let lambda_max = mid + term;
  let radius = 3.0 * sqrt(lambda_max);

  let min_pixel = out.center_2d - vec2<f32>(radius);
  let max_pixel = out.center_2d + vec2<f32>(radius);
  let tile_size = 16.0;
  let grid_width = camera.locW.w / tile_size;
  let grid_height = camera.atH.w / tile_size;

  let min_tile = vec2<i32>(max(vec2<f32>(0.0), floor(min_pixel / 16.0)));
  let max_tile = vec2<i32>(min(vec2<f32>(f32(grid_width - 1), f32(grid_height - 1)), floor(max_pixel / 16.0)));

  let bits_of_depth = bitcast<u32>(out.depth);

  for (var ty = min_tile.y; ty <= max_tile.y; ty++) {
    for (var tx = min_tile.x; tx <= max_tile.x; tx++) {
      let tile_id = u32(ty * i32(grid_width) + tx);
  
      let key = (tile_id << 16u) | (bits_of_depth >> 16u); 
  
      let global_key_idx = atomicAdd(&global_keys_counter, 1u);
  
      keysBuffer[global_key_idx].key = key;
      keysBuffer[global_key_idx].index = gs.index; 
    }
  }

  outputData[idx] = out; 
}
