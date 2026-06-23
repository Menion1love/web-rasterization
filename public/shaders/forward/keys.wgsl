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

@group(0) @binding(0) var<uniform> camera : camData;
@group(0) @binding(1) var<storage, read_write> inputData: array<rasterData>;
@group(0) @binding(2) var<storage, read_write> keysBuffer: array<u32>;
@group(0) @binding(3) var<storage, read_write> valuesBuffer: array<u32>;
@group(0) @binding(4) var<storage, read_write> global_keys_counter: atomic<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id : vec3<u32>,
  @builtin(local_invocation_index) local_index : u32
) {
  let workgroup_linear_id = workgroup_id.y * 256u + workgroup_id.x;
  let g_id = workgroup_linear_id * 64u + local_index; 

  let idx = g_id;

  if (idx >= arrayLength(&inputData)) {
    return; 
  }
  let val = atomicLoad(&global_keys_counter);

  let gs = inputData[idx];
  let tile_size = 16.0;

  let grid_width = camera.locW.w / tile_size;
  let grid_height = camera.atH.w / tile_size;
  let min_tile = gs.min_tile;
  let max_tile = gs.max_tile;

  let depth_bits = gs.depth;

  for (var ty = min_tile.y; ty <= max_tile.y; ty++) {
    for (var tx = min_tile.x; tx <= max_tile.x; tx++) {
      let tile_id = u32(ty) * u32(grid_width) + u32(tx);

      let key = (tile_id << 16u) | ((depth_bits >> 16u) & 0xFFFFu); 
  
      let global_key_idx = atomicAdd(&global_keys_counter, 1u);

      keysBuffer[global_key_idx] = key;
      valuesBuffer[global_key_idx] = idx; 
    }
  }
}
