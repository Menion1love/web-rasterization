@group(0) @binding(0) var<storage, read_write> density_stats: array<DensityStats>;

struct DensityStats {
  grad_accum: atomic<i32>,
  visible_count: atomic<u32>,
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= arrayLength(&density_stats)) { return; }
  atomicStore(&density_stats[idx].grad_accum, 0);
  atomicStore(&density_stats[idx].visible_count, 0u);
}