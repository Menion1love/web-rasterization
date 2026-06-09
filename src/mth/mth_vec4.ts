/***************************************************************
 * Copyright (C) 2026
 *    Computer Graphics Support Group of 30 Phys-Math Lyceum
 ***************************************************************/

/** FILE NAME   : mth_vec4.ts
 *  PURPOSE     : Gaussian splatting project.
 *  PROGRAMMER  : CGSG'An'2026.
 *                Timofey Hudyakov (TH4).
 *  LAST UPDATE : 02.06.2026
 * 
 *  No part of this file may be changed without agreement of
 *  Computer Graphics Support Group of 30 Phys-Math Lyceum
 */

import { mat4 } from "./mth_matr"

/** 
 * * Vector4d class 
 **/
class vec4 {
  /** #public parameters */
  public x: number = 0;
  public y: number = 0;
  public z: number = 0;
  public w: number = 0;
 
  /**
   * @info Class constructor
   * @param x: number
   * @param y: number
   * @param z: number
   * @param w: number
   */
  public constructor(x: number, y?: number, z?: number, w?: number) {
    this.x = x;
    if (y == undefined) this.y = x;
    else this.y = y;
    if (z == undefined) this.z = x;
    else this.z = z;
    if (w == undefined) this.w = x;
    else this.w = w;
  } /** End of constructor */

  /**
   * @info Evaluate vector length function
   * @returns none
   */
  public length2(): number {
    return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
  } /** End of 'length2' function */

  /**
   * @info Evaluate vector length function
   * @returns none
   */
  public length(): number {
    return Math.sqrt(this.length2());
  } /** End of 'length' function */

  /**
   * @info Vector addiction function
   * @param vector: vec4
   * @returns new vec4
   */
  public add(vector: vec4): vec4 {
    return new vec4(vector.x + this.x, vector.y + this.y, vector.z + this.z, vector.w + this.w);
  } /** End of 'add' function */

  /**
   * @info Vector subtracting function
   * @param vector: vec4
   * @returns new vec4
   */
  public sub(vector: vec4): vec4 {
    return new vec4(this.x - vector.x, this.y - vector.y, this.z - vector.z, this.w - vector.w);
  } /** End of 'sub' function */

  /**
   * @info Vector multipling by coordinates fucntion
   * @param vector: vec4
   * @returns new vec4
   */
  public mulVec(vector: vec4): vec4 {
    return new vec4(this.x * vector.x, this.y * vector.y, this.z * vector.z, this.w * vector.w);
  } /** End of 'mul' function */

  /**
   * @info Vector dividing by coordinates function
   * @param num: number
   * @returns new vec4
   */
  public div(num: number): vec4 {
    if (num == 0) return this;

    return new vec4(this.x / num, this.y / num, this.z / num, this.w / num);
  } /** End of 'mul' function */

  /**
   * @info Vector multipling by number function
   * @param num: number
   * @returns new vec4
   */
  public mulNum(num: number): vec4 {
    return new vec4(this.x * num, this.y * num, this.z * num, this.w * num);
  } /** End of 'mul' function */

  /**
   * @info Normalize vector function
   * @returns new vector
   */
  public normalize(): vec4 {
    let len: number = this.length();

    if (len == 0) return this;
    return this.div(len);
  } /** End of 'normalize' function */

  /**
   * @info normalizing vector function
   * @returns none
   */
  public normalizing(): void {
    let len: number = this.length();

    if (len == 0) return;

    this.x /= len;
    this.y /= len;
    this.z /= len;
    this.w /= len;
  } /** End of 'normalizing' function */

  /**
   * @info Dot product function
   * @param vector: vec4
   * @returns number
   */
  public dot(vector: vec4): number {
    return this.x * vector.x + this.y * vector.y + this.z * vector.z + this.w * vector.w;
  } /** End of 'dot' function */
  
  /**
   * @info Convert quaternion to matrix 4x4 function
   * @returns mat4
   */
  public q2m(): mat4 {
    const [x, y, z, w] = [this.x, this.y, this.z, this.w];

    const x2 = x + x;
    const y2 = y + y;
    const z2 = z + z;

    const xx = x * x2;
    const xy = x * y2;
    const xz = x * z2;
    
    const yy = y * y2;
    const yz = y * z2;
    const zz = z * z2;

    const wx = w * x2;
    const wy = w * y2;
    const wz = w * z2;

    return new mat4(
      1 - (yy + zz), 
      xy + wz, 
      xz - wy, 
      0,

      xy - wz, 
      1 - (xx + zz), 
      yz + wx, 
      0,

      xz + wy, 
      yz - wx, 
      1 - (xx + yy), 
      0,

      0, 
      0, 
      0, 
      1
    );
  } /** End of 'q2m' function */
} /** End of 'vec4' class */

/** EXPORTS */
export { vec4 };

/** END OF 'mth_vec4.ts' FILE */