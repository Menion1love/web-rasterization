struct gaussainData {
  position: vec3<f32>,
  opacity: f32,
  scale: vec3<f32>,
  index: u32,
  rotation: vec4<f32>,
  color: vec4<f32>,
}

struct DensityStats {
  grad_accum: atomic<i32>,
  visible_count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> inputData: array<gaussainData>;
@group(0) @binding(1) var<storage, read_write> adam_M: array<gaussainData>;
@group(0) @binding(2) var<storage, read_write> adam_V: array<gaussainData>;
@group(0) @binding(3) var<storage, read_write> alive_flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> density_stats: array<DensityStats>;
@group(0) @binding(5) var<storage, read_write> next_free_slot: atomic<u32>;
@group(0) @binding(6) var<uniform> params: DensityParams;

struct DensityParams {
  grad_threshold: f32,
  percent_dense: f32,
  prune_opacity_threshold: f32,
  capacity: u32,
  seed: u32,
  pad: vec3<f32>
}

fn quaternion_to_matrix(q_input: vec4<f32>) -> mat3x3<f32> {
  let q = normalize(q_input);
  let x = q.z; let y = q.y; let z = q.x; let w = q.w;
  let xx=x*x; let yy=y*y; let zz=z*z;
  let xy=x*y; let xz=x*z; let yz=y*z;
  let wx=w*x; let wy=w*y; let wz=w*z;
  return mat3x3<f32>(
    vec3<f32>(1.0-2.0*(yy+zz), 2.0*(xy+wz),       2.0*(xz-wy)),
    vec3<f32>(2.0*(xy-wz),     1.0-2.0*(xx+zz),   2.0*(yz+wx)),
    vec3<f32>(2.0*(xz+wy),     2.0*(yz-wx),       1.0-2.0*(xx+yy))
  );
}

fn hash(n: u32) -> u32 {
  var x = n;
  x = (x ^ 61u) ^ (x >> 16u);
  x = x + (x << 3u);
  x = x ^ (x >> 4u);
  x = x * 0x27d4eb2du;
  x = x ^ (x >> 15u);
  return x;
}

fn rand01(seed: u32) -> f32 {
  return f32(hash(seed)) / 4294967295.0;
}

fn rand_normal_vec3(idx: u32) -> vec3<f32> {
  let s = params.seed;
  let u1 = rand01(idx * 3u + 0u + s);
  let u2 = rand01(idx * 3u + 1u + s);
  let u3 = rand01(idx * 3u + 2u + s);
  return vec3<f32>(u1 - 0.5, u2 - 0.5, u3 - 0.5) * 2.0;
}

fn zero_gaussianData() -> gaussainData {
  var g: gaussainData;
  g.position = vec3<f32>(0.0);
  g.opacity = 0.0;
  g.scale = vec3<f32>(0.0);
  g.index = 0u;
  g.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  g.color = vec4<f32>(0.0);
  return g;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.capacity) { return; }
  if (alive_flags[idx] == 0u) { return; }

  let gs = inputData[idx];

  // --- PRUNE ---
  if (gs.opacity < params.prune_opacity_threshold) {
    alive_flags[idx] = 0u;
    return; 
  }

  let vis_count = atomicLoad(&density_stats[idx].visible_count);
  if (vis_count == 0u) { return; } 

  let avg_grad = f32(atomicLoad(&density_stats[idx].grad_accum)) / 1000.0 / f32(vis_count);
  let max_scale = max(max(gs.scale.x, gs.scale.y), gs.scale.z);

  if (avg_grad <= params.grad_threshold) {
    return; 
  }

  if (max_scale <= params.percent_dense) {
    // --- CLONE ---
    let new_idx = atomicAdd(&next_free_slot, 1u);
    if (new_idx >= params.capacity) { return; } 
    
    var new_gauss = gs;
    let offset = rand_normal_vec3(idx) * max_scale * 0.5;
    new_gauss.position = gs.position + offset;

    inputData[new_idx] = new_gauss;
    alive_flags[new_idx] = 1u;
    adam_M[new_idx] = zero_gaussianData();
    adam_V[new_idx] = zero_gaussianData();

  } else {
    // --- SPLIT ---
    let new_scale = gs.scale / 1.6;

    inputData[idx].scale = new_scale;
    adam_M[idx] = zero_gaussianData();
    adam_V[idx] = zero_gaussianData();

    let new_idx = atomicAdd(&next_free_slot, 1u);
    if (new_idx >= params.capacity) { return; }

    let R = quaternion_to_matrix(gs.rotation);
    let local_offset = rand_normal_vec3(idx + 999u) * new_scale;
    let world_offset = R * local_offset;

    var new_gauss = gs;
    new_gauss.scale = new_scale;
    new_gauss.position = gs.position + world_offset;

    inputData[new_idx] = new_gauss;
    alive_flags[new_idx] = 1u;
    adam_M[new_idx] = zero_gaussianData();
    adam_V[new_idx] = zero_gaussianData();
  }
}