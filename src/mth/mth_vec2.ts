/***************************************************************
 * Copyright (C) 2026
 *    Computer Graphics Support Group of 30 Phys-Math Lyceum
 ***************************************************************/

/** FILE NAME   : mth_vec2.ts
 *  PURPOSE     : Gaussian splatting project.
 *  PROGRAMMER  : CGSG'An'2026.
 *                Timofey Hudyakov (TH4).
 *  LAST UPDATE : 02.06.2026
 * 
 *  No part of this file may be changed without agreement of
 *  Computer Graphics Support Group of 30 Phys-Math Lyceum
 */

/** 
 * * Vector2d class 
 **/
class vec2 {
  /** #public parameters */
  public x: number = 0;
  public y: number = 0;
 
  /**
   * @info Class constructor
   * @param x: number
   * @param y: number
   */
  public constructor(x: number, y?: number) {
    this.x = x;
    if (y == undefined) this.y = x;
    else this.y = y;
  } /** End of constructor */

  /**
   * @info Evaluate vector length function
   * @returns none
   */
  public length2(): number {
    return this.x * this.x + this.y * this.y;
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
   * @param vector: vec2
   * @returns new vec2
   */
  public add(vector: vec2): vec2 {
    return new vec2(vector.x + this.x, vector.y + this.y);
  } /** End of 'add' function */

  /**
   * @info Vector subtracting function
   * @param vector: vec2
   * @returns new vec2
   */
  public sub(vector: vec2): vec2 {
    return new vec2(this.x - vector.x, this.y - vector.y);
  } /** End of 'sub' function */

  /**
   * @info Vector multipling by coordinates fucntion
   * @param vector: vec2
   * @returns new vec2
   */
  public mulVec(vector: vec2): vec2 {
    return new vec2(this.x * vector.x, this.y * vector.y);
  } /** End of 'mul' function */

  /**
   * @info Vector dividing by coordinates function
   * @param num: number
   * @returns new vec2
   */
  public div(num: number): vec2 {
    if (num == 0) return this;

    return new vec2(this.x / num, this.y / num);
  } /** End of 'mul' function */

  /**
   * @info Vector multipling by number function
   * @param num: number
   * @returns new vec2
   */
  public mulNum(num: number): vec2 {
    return new vec2(this.x * num, this.y * num);
  } /** End of 'mul' function */

  /**
   * @info Normalize vector function
   * @returns new vector
   */
  public normalize(): vec2 {
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
  } /** End of 'normalizing' function */
} /** End of 'vec2' class */

/** EXPORTS */
export { vec2 };

/** END OF 'mth_vec2.ts' FILE */