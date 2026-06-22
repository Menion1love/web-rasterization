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
  opacity: f32,
  scale: vec3<f32>,
  index: u32,
  rotation: vec4<f32>,
  color: vec4<f32>,
}

struct SingleGaussianGrad {
  opacity: i32, 
  center_x: i32,
  center_y: i32,
  cov_a: i32,  
  cov_b: i32,  
  cov_c: i32,  
  color_r: i32,
  color_g: i32,
  color_b: i32,
  
  pad0: i32,
  pad1: i32,
  pad2: i32,
}

struct GaussianGradients {
  grads: array<SingleGaussianGrad>, 
}

@group(0) @binding(0) var<uniform> camera : camData;
@group(0) @binding(1) var<storage, read_write> inputData: array<gaussainData>;
@group(0) @binding(2) var<storage, read_write> global_grads: GaussianGradients;
@group(0) @binding(3) var<storage, read_write> adam_M: array<gaussainData>;
@group(0) @binding(4) var<storage, read_write> adam_V: array<gaussainData>;
@group(0) @binding(5) var<uniform> iterator: u32;

fn quaternion_to_matrix(q_input: vec4<f32>) -> mat3x3<f32> {
  let q = normalize(q_input);
  
  let x = q.z;
  let y = q.y;
  let z = q.x;
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

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3u,
  @builtin(workgroup_id) workgroup_id: vec3u,
  @builtin(global_invocation_id) global_id: vec3u
) {
  let idx = global_id.x;
  let g_id = global_id.x;
  let gs = inputData[idx];

  let scale = 1000.0; 

  let dL_dopacity  = f32(global_grads.grads[g_id].opacity) / scale;
  let dL_dcolor_r  = f32(global_grads.grads[g_id].color_r) / scale;
  let dL_dcolor_g  = f32(global_grads.grads[g_id].color_g) / scale;
  let dL_dcolor_b  = f32(global_grads.grads[g_id].color_b) / scale;
  let dL_dcolor    = vec3f(dL_dcolor_r, dL_dcolor_g, dL_dcolor_b);

  let dL_dcenter_2d = vec2f(
      f32(global_grads.grads[g_id].center_x) / scale,
      f32(global_grads.grads[g_id].center_y) / scale
  );
  let dL_dcov_2d = vec3f(
      f32(global_grads.grads[g_id].cov_a) / scale,
      f32(global_grads.grads[g_id].cov_b) / scale,
      f32(global_grads.grads[g_id].cov_c) / scale
  );

  let t = camera.view * vec4<f32>(gs.position, 1.0);

  let half_w = f32(camera.locW.w) * 0.5;
  let half_h = f32(camera.atH.w) * 0.5;

  let fx = camera.proj[0][0] * half_w;
  let fy = camera.proj[1][1] * half_h;

  let tz2 = t.z * t.z;
  let J_00 = fx / t.z;
  let J_01 = 0.0;
  let J_02 = -(t.x * fx) / tz2;

  let J_10 = 0.0;
  let J_11 = -fy / t.z;
  let J_12 = (t.y * fy) / tz2;

  var dL_dxyz_camera = vec3f(0.0);
  dL_dxyz_camera.x = dL_dcenter_2d.x * J_00 + dL_dcenter_2d.y * J_10;
  dL_dxyz_camera.y = dL_dcenter_2d.x * J_01 + dL_dcenter_2d.y * J_11;
  dL_dxyz_camera.z = dL_dcenter_2d.x * J_02 + dL_dcenter_2d.y * J_12;

  let R_cam = mat3x3f(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
  var dL_dxyz_world = transpose(R_cam) * dL_dxyz_camera;

  let dL_da = dL_dcov_2d.x;
  let dL_db = dL_dcov_2d.y;
  let dL_dc = dL_dcov_2d.z;

  let dL_dSigma2D = mat3x3f(
    vec3<f32>(dL_da, dL_db * 0.5, 0.0),
    vec3<f32>(dL_db * 0.5, dL_dc, 0.0),
    vec3<f32>(0.0, 0.0, 0.0),
  );

  let J = mat3x3<f32>(
    vec3<f32>(J_00, 0, J_02),
    vec3<f32>(0, J_11, J_12),
    vec3<f32>(0, 0, 0.0),
  );

  let w = mat3x3<f32>(
    camera.view[0].xyz,
    camera.view[1].xyz,
    camera.view[2].xyz
  );

  let T = J * w;

  let dL_dSigma3D: mat3x3f = transpose(T) * dL_dSigma2D * T;

  let R = quaternion_to_matrix(gs.rotation);
  let s = vec3<f32>(gs.scale);
  var M: mat3x3<f32>;
  M[0] = R[0] * s.z; 
  M[1] = R[1] * s.y; 
  M[2] = R[2] * s.x; 

  let dL_dM = 2.0 * dL_dSigma3D * M;

  let dL_dS_mat = transpose(R) * dL_dM;

  var dL_dScale = vec3f(0.0);
  dL_dScale.x = dL_dS_mat[2][2];
  dL_dScale.y = dL_dS_mat[1][1];
  dL_dScale.z = dL_dS_mat[0][0];

  // dL_dScale = dL_dScale.zyx;

  let dL_dR = dL_dM * mat3x3f(vec3f(s.z,0,0), vec3f(0,s.y,0), vec3f(0,0,s.x)); 

  let q = gs.rotation;
  let qw = q.w; let qx = q.z; let qy = q.y; let qz = q.x; 

  let dL_dz_local = 2.0 * (
    -2.0 * qz * dL_dR[0][0] - qw * dL_dR[1][0] + qx * dL_dR[2][0]
    + qw * dL_dR[0][1] - 2.0 * qz * dL_dR[1][1] + qy * dL_dR[2][1]
    + qx * dL_dR[0][2] + qy * dL_dR[1][2]
  );

  let dL_dy_local = 2.0 * (
    -2.0 * qy * dL_dR[0][0] + qx * dL_dR[1][0] + qw * dL_dR[2][0]
    + qx * dL_dR[0][1] + qz * dL_dR[2][1]
    - qw * dL_dR[0][2] + qz * dL_dR[1][2] - 2.0 * qy * dL_dR[2][2]
  );

  let dL_dx_local = 2.0 * (
    qy * dL_dR[1][0] + qz * dL_dR[2][0]
    + qy * dL_dR[0][1] - 4.0 * qx * dL_dR[1][1] - qw * dL_dR[2][1]
    + qz * dL_dR[0][2] + qw * dL_dR[1][2] - 4.0 * qx * dL_dR[2][2]
  );

  let dL_dw_local = 2.0 * (
    -qz * dL_dR[1][0] + qy * dL_dR[2][0]
    + qz * dL_dR[0][1] - qx * dL_dR[2][1]
    - qy * dL_dR[0][2] + qx * dL_dR[1][2]
  );

  var dL_dq = vec4f(0.0);
  dL_dq.x = dL_dz_local; 
  dL_dq.y = dL_dy_local; 
  dL_dq.z = dL_dx_local; 
  dL_dq.w = dL_dw_local; 

  // dL_dq = -1.0 * dL_dq; 


  let beta1 = 0.9;
  let beta2 = 0.999;
  let eps = 1e-8;
  let it = f32(iterator); 
  let bias_corr1 = 1.0 - pow(0.9, it);
  let bias_corr2 = 1.0 - pow(0.999, it);
  
  let lr_xyz = 0.0016;
  let lr_rotation = 0.004;
  let lr_color = 0.001;
  let lr_opacity = 0.05;
  let lr_scale = 0.008;

  let current_linear_scale = inputData[g_id].scale;
  let current_log_scale = log(current_linear_scale + vec3f(1e-8));
  let dL_dLogScale = dL_dScale * current_linear_scale;

  adam_M[g_id].scale = beta1 * adam_M[g_id].scale + (1.0 - beta1) * dL_dLogScale;
  adam_V[g_id].scale = beta2 * adam_V[g_id].scale + (1.0 - beta2) * (dL_dLogScale * dL_dLogScale);

  var new_log_scale = current_log_scale - lr_scale * (adam_M[g_id].scale / bias_corr1) / (sqrt(adam_V[g_id].scale / bias_corr2) + eps);
  inputData[g_id].scale = clamp(exp(new_log_scale), vec3f(1e-5), vec3f(1.0));

  adam_M[g_id].rotation = beta1 * adam_M[g_id].rotation + (1.0 - beta1) * dL_dq;
  adam_V[g_id].rotation = beta2 * adam_V[g_id].rotation + (1.0 - beta2) * (dL_dq * dL_dq);

  var new_rot = inputData[g_id].rotation - lr_rotation * (adam_M[g_id].rotation / bias_corr1) / (sqrt(adam_V[g_id].rotation / bias_corr2) + eps);
  inputData[g_id].rotation = normalize(new_rot); 

  adam_M[g_id].color = vec4f(beta1 * adam_M[g_id].color.xyz + (1.0 - beta1) * dL_dcolor, 0.0);
  adam_V[g_id].color = vec4f(beta2 * adam_V[g_id].color.xyz + (1.0 - beta2) * (dL_dcolor * dL_dcolor), 0.0);

  let hat_M_col = adam_M[g_id].color.xyz / bias_corr1;
  let hat_V_col = adam_V[g_id].color.xyz / bias_corr2;

  var new_color_xyz = inputData[g_id].color.xyz - lr_color * hat_M_col / (sqrt(hat_V_col) + eps);
  inputData[g_id].color = vec4f(clamp(new_color_xyz, vec3f(0.0), vec3f(1.0)), 1.0);
  
  let current_op = clamp(inputData[g_id].opacity, 1e-5, 0.995);
  let current_logit_op = log(current_op / (1.0 - current_op)); 
  
  let dL_dLogitOp = dL_dopacity * current_op * (1.0 - current_op);

  adam_M[g_id].opacity = beta1 * adam_M[g_id].opacity + (1.0 - beta1) * dL_dLogitOp;
  adam_V[g_id].opacity = beta2 * adam_V[g_id].opacity + (1.0 - beta2) * (dL_dLogitOp * dL_dLogitOp);
  
  let hat_M_op = adam_M[g_id].opacity / bias_corr1;
  let hat_V_op = adam_V[g_id].opacity / bias_corr2;
  
  var new_logit_op = current_logit_op - lr_opacity * hat_M_op / (sqrt(hat_V_op) + eps);
  
  inputData[g_id].opacity = 1.0 / (1.0 + exp(-new_logit_op));

  adam_M[g_id].position = beta1 * adam_M[g_id].position + (1.0 - beta1) * dL_dxyz_world.xyz;
  adam_V[g_id].position = beta2 * adam_V[g_id].position + (1.0 - beta2) * (dL_dxyz_world.xyz * dL_dxyz_world.xyz);

  let hat_M_xyz = adam_M[g_id].position / bias_corr1;
  let hat_V_xyz = adam_V[g_id].position / bias_corr2;

  inputData[g_id].position -= lr_xyz * hat_M_xyz / (sqrt(hat_V_xyz) + eps);
}