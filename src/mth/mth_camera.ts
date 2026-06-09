/***************************************************************
 * Copyright (C) 2026
 *    Computer Graphics Support Group of 30 Phys-Math Lyceum
 ***************************************************************/

/** FILE NAME   : mth_camera.ts
 *  PURPOSE     : Gaussian splatting project.
 *  PROGRAMMER  : CGSG'An'2026.
 *                Timofey Hudyakov (TH4).
 *  LAST UPDATE : 02.06.2026
 * 
 *  No part of this file may be changed without agreement of
 *  Computer Graphics Support Group of 30 Phys-Math Lyceum
 */

import { mat4 } from './mth_matr';
import { vec3 } from './mth_vec3';

/** Convert radians to degrees */
const r2D = (rad: number): number => rad * 180 / Math.PI;


/** 
 * * Camera class 
 **/
class camera {
  /** #public parameters */
  public loc: vec3 = new vec3(0, 0, -5);
  public at: vec3 = new vec3(0, 0, 0);
  public up: vec3 = new vec3(0, 1, 0);
  public right: vec3 = new vec3(1, 0, 0);
  public dir: vec3 = new vec3(0, 0, 1);

  public projDist: number = 0.1;
  public projFar: number = 1000;
  public projSize: number = 0.1;
  public wp: number = 0.1;
  public hp: number = 0.1;
  public frameW: number = 1024;
  public frameH: number = 680;
  public cosT: number = 0;
  public sinT: number = 0;
  public cosP: number = 0;
  public sinP: number = 0;
  public plen: number = 0;
  public azimuth: number = 0;
  public elevator: number = 0;
  public deltaAzimuth: number = 0;
  public deltaElevator: number = 0;
  public deltaDist: number = 0;
  public dist: number = 0;
  public proj!: mat4;
  public view!: mat4;
  public vp!: mat4;

  /**
   * @info Class constructor
   * @param frameW: number
   * @param frameH: number
   */
  public constructor(frameW: number, frameH: number) {
    this.frameW = frameW;
    this.frameH = frameH;
  } /** End of constructor */

  /**
   * @info Set new projection function
   * @returns none
   */
  public setProj(): void {
    let wp = this.projSize;
    let hp = this.projSize;

    if (this.frameW > this.frameH) wp *= this.frameW / this.frameH;
    else hp *= this.frameH / this.frameW;

    this.wp = wp;
    this.hp = hp;
    this.proj = mat4.frustum(
      -wp / 2,
      wp / 2,
      -hp / 2,
      hp / 2,
      this.projSize,
      this.projFar,
    );

  } /** End of 'setProj' function */

  /**
   * @info Set orientation function
   * @returns none
   */
  public setOrientation(): void {
    this.dist = this.at.sub(this.loc).length();
    this.cosT = (this.loc.y - this.at.y) / this.dist;
    this.sinT = Math.sqrt(1 - this.cosT * this.cosT);
    this.plen = this.dist * this.sinT;
    this.cosP = (this.loc.z - this.at.z) / this.plen;
    this.sinP = (this.loc.x - this.at.x) / this.plen;
    this.azimuth = r2D(Math.atan2(this.sinP, this.cosP));
    this.elevator = r2D(Math.atan2(this.sinT, this.cosT));
    this.deltaAzimuth = 0;
    this.deltaElevator = 0;
    this.deltaDist = 0;
  } /** End of 'setOrientation' function */

  /**
   * @info Set default camera function
   * @returns none
   */
  public setDefault(): void {
    this.set(new vec3(0, 0, -5), new vec3(0, 0, 0));
  } /** End of 'setDefault' function */

  /**
   * @info Set location, at point & up function
   * @param loc: mth.vec4
   * @param at: mth.vec4
   * @param up: mth.vec4
   * @returns none
   */
  public  set(
    loc: vec3,
    at: vec3,
    up: vec3 = new vec3(0, 1, 0),
  ): void {
    this.loc = loc;
    this.at = at;
    this.up = up;
    this.view = mat4.view(loc, at, up);

    this.setProj();
    
    // Calculate camera vectors directly
    this.dir = this.at.sub(this.loc).normalize();
    this.right = this.dir.cross(up).normalize();
    this.up = this.right.cross(this.dir).normalize();
    this.vp = this.proj.multiply(this.view);
  } /** End of 'set' function */
} /** End of 'camera' class */

/** EXPORTS */
export { camera };

/** END OF 'mth_camera.ts' FILE */