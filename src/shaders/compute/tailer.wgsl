struct tailData {
  start_idx: u32,
  end_idx: u32,
}

@group(0) @binding(0) var<storage, read_write> keysBuffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> tail_ranges: array<tailData>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>, @builtin(global_invocation_id) global_id : vec3<u32>) {
  let idx = global_id.x;
  let size = arrayLength(&keysBuffer);

  let current_key = keysBuffer[idx];
  let current_tile = current_key >> 16u; 

  if (idx == 0u) {
      tail_ranges[current_tile].start_idx = 0u; 
  } else {
      let prev_key = keysBuffer[idx - 1u];
      let prev_tile = prev_key >> 16u;

      if (current_tile != prev_tile) {
          tail_ranges[current_tile].start_idx = idx;
          tail_ranges[prev_tile].end_idx = idx;
      }
  }

  if (idx == size - 1u) {
    tail_ranges[current_tile].end_idx = size;
  }
}

