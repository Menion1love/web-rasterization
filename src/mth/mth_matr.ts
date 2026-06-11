/***************************************************************
 * Copyright (C) 2026
 *    Computer Graphics Support Group of 30 Phys-Math Lyceum
 ***************************************************************/

/** FILE NAME   : mth_matr.ts
 *  PURPOSE     : Gaussian splatting project.
 *  PROGRAMMER  : CGSG'An'2026.
 *                Timofey Hudyakov (TH4).
 *  LAST UPDATE : 02.06.2026
 * 
 *  No part of this file may be changed without agreement of
 *  Computer Graphics Support Group of 30 Phys-Math Lyceum
 */

import { vec3 } from './mth_vec3';
import { vec4 } from './mth_vec4';

/** Convert degrees to radians */
const d2R = (rad: number): number => rad * Math.PI / 180;

/** 
 * * Matrix4x4 class 
 **/
class mat4 {
  /** Matrix data as flat array (column-major order) */
  public m: number[] = new Array(16);

  /**
   * @info Class constructor
   * @param m00-m33: number - matrix elements
   */
  public constructor(
    m00?: number, m01?: number, m02?: number, m03?: number,
    m10?: number, m11?: number, m12?: number, m13?: number,
    m20?: number, m21?: number, m22?: number, m23?: number,
    m30?: number, m31?: number, m32?: number, m33?: number
  ) {
    if (m00 !== undefined) {
      this.m = [
        m00, m01 || 0, m02 || 0, m03 || 0,
        m10 || 0, m11 || 0, m12 || 0, m13 || 0,
        m20 || 0, m21 || 0, m22 || 0, m23 || 0,
        m30 || 0, m31 || 0, m32 || 0, m33 || 0
      ];
    } else {
      this.identity();
    }
  } /** End of constructor */

  /**
   * @info Set matrix to identity
   * @returns this matrix
   */
  public identity(): mat4 {
    this.m.fill(0);
    this.m[0] = this.m[5] = this.m[10] = this.m[15] = 1;
    return this;
  } /** End of 'identity' function */

  /**
   * @info Create identity matrix
   * @returns new identity matrix
   */
  public static identity(): mat4 {
    return new mat4(
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    );
  } /** End of 'identity' function */

  /**
   * @info Create rotation matrix around X axis
   * @param angle: number - angle in degrees
   * @returns new rotation matrix
   */
  public static rotateX(angle: number): mat4 {
    const c = Math.cos(d2R(angle));
    const s = Math.sin(d2R(angle));
    return new mat4(
      1,  0,  0, 0,
      0,  c,  s, 0,
      0, -s,  c, 0,
      0,  0,  0, 1
    );
  } /** End of 'rotateX' function */

  /**
   * @info Create rotation matrix around Y axis
   * @param angle: number - angle in degrees
   * @returns new rotation matrix
   */
  public static rotateY(angle: number): mat4 {
    const c = Math.cos(d2R(angle));
    const s = Math.sin(d2R(angle));
    return new mat4(
      c, 0, -s, 0,
      0, 1,  0, 0,
      s, 0,  c, 0,
      0, 0,  0, 1
    );
  } /** End of 'rotateY' function */

  /**
   * @info Create rotation matrix around Z axis
   * @param angle: number - angle in radians
   * @returns new rotation matrix
   */
  public static rotateZ(angle: number): mat4 {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new mat4(
       c, s, 0, 0,
      -s, c, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1
    );
  } /** End of 'rotateZ' function */

  /**
   * @info Create rotation matrix around arbitrary vector
   * @param axis: vec3 - rotation axis
   * @param angle: number - angle in radians
   * @returns new rotation matrix
   */
  public static rotate(axis: vec3, angle: number): mat4 {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const t = 1 - c;
    const n = axis.normalize();
    
    return new mat4(
      t * n.x * n.x + c, t * n.x * n.y + s * n.z, t * n.x * n.z - s * n.y, 0,
      t * n.x * n.y - s * n.z, t * n.y * n.y + c, t * n.y * n.z + s * n.x, 0,
      t * n.x * n.z + s * n.y, t * n.y * n.z - s * n.x, t * n.z * n.z + c, 0,
      0, 0, 0, 1
    );
  } /** End of 'rotate' function */

  /**
   * @info Create scale matrix
   * @param scale: vec3 - scale factors
   * @returns new scale matrix
   */
  public static scale(scale: vec3): mat4 {
    return new mat4(
      scale.x, 0, 0, 0,
      0, scale.y, 0, 0,
      0, 0, scale.z, 0,
      0, 0, 0, 1
    );
  } /** End of 'scale' function */

  /**
   * @info Create translation matrix
   * @param translation: vec3 - translation vector
   * @returns new translation matrix
   */
  public static translate(translation: vec3): mat4 {
    return new mat4(
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      translation.x, translation.y, translation.z, 1
    );
  } /** End of 'translate' function */

  /**
   * @info Transpose matrix
   * @returns new transposed matrix
   */
  public transpose(): mat4 {
    const result = new mat4();
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        result.m[i * 4 + j] = this.m[j * 4 + i];
      }
    }
    return result;
  } /** End of 'transpose' function */

  /**
   * @info Transform vector by matrix
   * @param vector: vec4 - vector to transform
   * @returns new transformed vector
   */
  public transform(vector: vec4): vec4 {
    return new vec4(
      this.m[0] * vector.x + this.m[4] * vector.y + this.m[8] * vector.z + this.m[12] * vector.w,
      this.m[1] * vector.x + this.m[5] * vector.y + this.m[9] * vector.z + this.m[13] * vector.w,
      this.m[2] * vector.x + this.m[6] * vector.y + this.m[10] * vector.z + this.m[14] * vector.w,
      this.m[3] * vector.x + this.m[7] * vector.y + this.m[11] * vector.z + this.m[15] * vector.w
    );
  } /** End of 'transform' function */

  /**
   * @info Transform point by matrix
   * @param point: vec3 - point to transform
   * @returns new transformed point
   */
  public transformPoint(point: vec3): vec3 {
    return new vec3(
      this.m[0] * point.x + this.m[4] * point.y + this.m[8] * point.z + this.m[12],
      this.m[1] * point.x + this.m[5] * point.y + this.m[9] * point.z + this.m[13],
      this.m[2] * point.x + this.m[6] * point.y + this.m[10] * point.z + this.m[14]
    );
  } /** End of 'TransformPoint' function */

  /**
   * @info Create orthographic projection matrix
   * @param left: number
   * @param right: number
   * @param bottom: number
   * @param top: number
   * @param near: number
   * @param far: number
   * @returns new ortho matrix
   */
  public static ortho(left: number, right: number, bottom: number, top: number, near: number, far: number): mat4 {
    return new mat4(
      2 / (right - left), 0, 0, 0,
      0, 2 / (top - bottom), 0, 0,
      0, 0, -2 / (far - near), 0,
      -(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1
    );
  } /** End of 'ortho' function */

  /**
   * @info Create frustum projection matrix
   * @param left: number
   * @param right: number
   * @param bottom: number
   * @param top: number
   * @param near: number
   * @param far: number
   * @returns new frustum matrix
   */
  public static frustum(left: number, right: number, bottom: number, top: number, near: number, far: number): mat4 {
    return new mat4(
      2 * near / (right - left), 0, 0, 0,
      0, 2 * near / (top - bottom), 0, 0,
      (right + left) / (right - left), (top + bottom) / (top - bottom), far / (far - near), 1,
      0, 0, -far * near / (far - near),  0
    );
  } /** End of 'frustum' function */

  /**
   * @info Create view matrix
   * @param eye: vec3 - camera position
   * @param center: vec3 - look at point
   * @param up: vec3 - up vector
   * @returns new view matrix
   */
  public static view(eye: vec3, center: vec3, up: vec3): mat4 {
    const f = center.sub(eye).normalize();
    const s = f.cross(up).normalize();
    const u = s.cross(f);
    
    return new mat4(
      s.x, u.x, f.x, 0,
      s.y, u.y, f.y, 0,
      s.z, u.z, f.z, 0,
      -s.dot(eye), -u.dot(eye), -f.dot(eye), 1
    );
  } /** End of 'view' function */

  /**
   * @info Multiply matrices
   * @param other: mat4 - matrix to multiply with
   * @returns new result matrix
   */
  public multiply(other: mat4): mat4 {
    const a = this.m, b = other.m;
    return new mat4(
      a[0] * b[0] + a[4] * b[1] + a[8] * b[2] + a[12] * b[3],
      a[1] * b[0] + a[5] * b[1] + a[9] * b[2] + a[13] * b[3],
      a[2] * b[0] + a[6] * b[1] + a[10] * b[2] + a[14] * b[3],
      a[3] * b[0] + a[7] * b[1] + a[11] * b[2] + a[15] * b[3],
      a[0] * b[4] + a[4] * b[5] + a[8] * b[6] + a[12] * b[7],
      a[1] * b[4] + a[5] * b[5] + a[9] * b[6] + a[13] * b[7],
      a[2] * b[4] + a[6] * b[5] + a[10] * b[6] + a[14] * b[7],
      a[3] * b[4] + a[7] * b[5] + a[11] * b[6] + a[15] * b[7],
      a[0] * b[8] + a[4] * b[9] + a[8] * b[10] + a[12] * b[11],
      a[1] * b[8] + a[5] * b[9] + a[9] * b[10] + a[13] * b[11],
      a[2] * b[8] + a[6] * b[9] + a[10] * b[10] + a[14] * b[11],
      a[3] * b[8] + a[7] * b[9] + a[11] * b[10] + a[15] * b[11],
      a[0] * b[12] + a[4] * b[13] + a[8] * b[14] + a[12] * b[15],
      a[1] * b[12] + a[5] * b[13] + a[9] * b[14] + a[13] * b[15],
      a[2] * b[12] + a[6] * b[13] + a[10] * b[14] + a[14] * b[15],
      a[3] * b[12] + a[7] * b[13] + a[11] * b[14] + a[15] * b[15]
    );
  } /** End of 'multiply' function */

  /**
   * @info Calculate matrix inverse
   * @returns new inverted matrix or null if not invertible
   */
  public inverse(): mat4 | null {
    const m = this.m;
    const inv = new Array(16);

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    const det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (Math.abs(det) < 1e-10) return null;

    const invDet = 1.0 / det;
    for (let i = 0; i < 16; i++) {
      inv[i] *= invDet;
    }

    return new mat4(
      inv[0], inv[1], inv[2], inv[3],
      inv[4], inv[5], inv[6], inv[7],
      inv[8], inv[9], inv[10], inv[11],
      inv[12], inv[13], inv[14], inv[15]
    );
  } /** End of 'inverse' function */
} /** End of 'mat4' class */

/** EXPORTS */
export { mat4 };

/** END OF 'mth_matr.ts' FILE */