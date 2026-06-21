struct tailData {
  start_idx: u32,
  end_idx: u32,
}

@group(0) @binding(0) var<storage, read_write> keysBuffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> tail_ranges: array<tailData>;
@group(0) @binding(2) var<storage, read_write> global_keys_counter: u32;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>, @builtin(global_invocation_id) global_id : vec3<u32>) {
  let idx = global_id.x;
  let size = arrayLength(&keysBuffer);
  let first_key = size - global_keys_counter;

  if (idx >= size) {
    return; 
  }

  if (idx < first_key) {
    return;
  }

  let current_key = keysBuffer[idx];
  let current_tile = current_key >> 16u; 
  
  if (idx == first_key) {
    tail_ranges[current_tile].start_idx = first_key; 
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

