/***************************************************************
 * Copyright (C) 2026
 *    Computer Graphics Support Group of 30 Phys-Math Lyceum
 ***************************************************************/

/** FILE NAME   : mth_vec3.ts
 *  PURPOSE     : Gaussian splatting project.
 *  PROGRAMMER  : CGSG'An'2026.
 *                Timofey Hudyakov (TH4).
 *  LAST UPDATE : 02.06.2026
 * 
 *  No part of this file may be changed without agreement of
 *  Computer Graphics Support Group of 30 Phys-Math Lyceum
 */

/** 
 * * Vector3d class 
 **/
class vec3 {
  /** #public parameters */
  public x: number = 0;
  public y: number = 0;
  public z: number = 0;
 
  /**
   * @info Class constructor
   * @param x: number
   * @param y: number
   * @param z: number
   */
  public constructor(x: number, y?: number, z?: number) {
    this.x = x;
    if (y == undefined) this.y = x;
    else this.y = y;
    if (z == undefined) this.z = x;
    else this.z = z;
  } /** End of constructor */

  /**
   * @info Evaluate vector length function
   * @returns none
   */
  public length2(): number {
    return this.x * this.x + this.y * this.y + this.z * this.z;
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
   * @param vector: vec3
   * @returns new vec3
   */
  public add(vector: vec3): vec3 {
    return new vec3(vector.x + this.x, vector.y + this.y, vector.z + this.z);
  } /** End of 'add' function */

  /**
   * @info Vector subtracting function
   * @param vector: vec3
   * @returns new vec3
   */
  public sub(vector: vec3): vec3 {
    return new vec3(this.x - vector.x, this.y - vector.y, this.z - vector.z);
  } /** End of 'sub' function */

  /**
   * @info Vector multipling by coordinates fucntion
   * @param vector: vec3
   * @returns new vec3
   */
  public mulVec(vector: vec3): vec3 {
    return new vec3(this.x * vector.x, this.y * vector.y, this.z * vector.z);
  } /** End of 'mul' function */
    
  /**
   * @info Vector dividing by coordinates function
   * @param num: number
   * @returns new vec3
   */
  public div(num: number): vec3 {
    if (num == 0) return this;

    return new vec3(this.x / num, this.y / num, this.z / num);
  } /** End of 'mul' function */

  /**
   * @info Vector multipling by number function
   * @param num: number
   * @returns new vec3
   */
  public mulNum(num: number): vec3 {
    return new vec3(this.x * num, this.y * num, this.z * num);
  } /** End of 'mul' function */

  /**
   * @info Normalize vector function
   * @returns new vector
   */
  public normalize(): vec3 {
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
  } /** End of 'normalizing' function */

  /**
   * @info Dot product function
   * @param vector: vec3
   * @returns number
   */
  public dot(vector: vec3): number {
    return this.x * vector.x + this.y * vector.y + this.z * vector.z;
  } /** End of 'dot' function */

  /**
   * @info Cross product function
   * @param vector: vec3
   * @returns new vec3
   */
  public cross(vector: vec3): vec3 {
    return new vec3(
      this.y * vector.z - this.z * vector.y,
      this.z * vector.x - this.x * vector.z,
      this.x * vector.y - this.y * vector.x
    );
  } /** End of 'cross' function */
} /** End of 'vec3' class */

/** EXPORTS */
export { vec3 };

/** END OF 'mth_vec3.ts' FILE */