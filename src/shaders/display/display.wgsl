struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
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
  
  output.uv = pos[vertexIndex] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
  
  return output;
}

// fragment shader
@group(0) @binding(0) var outTexture: texture_2d<f32>;
@group(0) @binding(1) var outSampler: sampler;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  return textureSample(outTexture, outSampler, input.uv);
}
