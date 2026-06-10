
struct VertexOutput {
  @location(0) color: vec4<f32>,
  @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  
  var output: VertexOutput;
  output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
  output.color = vec4<f32>(1.0);
  return output;
}

// fragment shader
@group(0) @binding(0) var outTexture: texture_2d<f32>;
@group(0) @binding(1) var outSampler: sampler;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = vec2<f32>(pos.x / 2.0 + 0.5, 1.0 - pos.y / 2.0 - 0.5);
  return textureSample(outTexture, outSampler, uv);
}
