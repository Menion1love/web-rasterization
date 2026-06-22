/***************************************************************
 * Copyright (C) 2026
 *    Computer Graphics Support Group of 30 Phys-Math Lyceum
 ***************************************************************/

/** FILE NAME   : control.ts
 *  PURPOSE     : Gaussian splatting project.
 *  PROGRAMMER  : CGSG'An'2026.
 *                Timofey Hudyakov (TH4).
 *  LAST UPDATE : 09.06.2026
 * 
 *  No part of this file may be changed without agreement of
 *  Computer Graphics Support Group of 30 Phys-Math Lyceum
 */

import { input } from "./input";
import { timer } from "./timer";
import { camera } from "../mth/mth_camera";
import { vec3 } from "../mth/mth_vec3";

/** 
 * * Camera control class 
 **/
class control {
  /** #private parameters */
  private targetAzimuth: number = 0;
  private targetElevator: number = 90;
  private targetDist: number = 10;
  private targetAt: vec3 = new vec3(0, 0, 0);
  private smoothFactor: number = 0.08;
  private moveSpeed: number = 1.0;
  private rotateSpeed: number = 1.0;
  private cameraMode: 'thirdPerson' | 'firstPerson' = 'thirdPerson';
  private yaw: number = 0;
  private pitch: number = 0;
  
  /** #public parameters */
  public cam!: camera; // Render camera
  public onCameraMove: (() => void) | null = null; // Callback for camera movement
  private lastCameraPos: vec3 = new vec3(0, 0, 0);
  private lastCameraAt: vec3 = new vec3(0, 0, 0);
  private movementThreshold: number = 0.01; // Minimum movement to trigger reset  

  /**
   * @info Controls constructor
   * @returns none
   */
  public constructor() {
    this.cam = new camera(1024, 768);
    this.cam.set(new vec3(0, 0, 5), new vec3(0, 0, 0), new vec3(0, 1, 0));
    this.cam.setOrientation();
    this.targetAzimuth = this.cam.azimuth || 0;
    this.targetElevator = this.cam.elevator || 90;
    this.targetDist = this.cam.dist || 10;
    this.targetAt = this.cam.at || new vec3(0, 0, 0);
    
    window.addEventListener('updateSceneParams', (e: any) => {
      const params = e.detail;
      this.moveSpeed = params.cameraMoveSpeed || 1.0;
      this.rotateSpeed = params.cameraRotateSpeed || 1.0;
      const newMode = params.cameraMode || 'thirdPerson';
      if (newMode !== this.cameraMode) {
        const oldMode = this.cameraMode;
        this.cameraMode = newMode;
        if (newMode === 'firstPerson') {
          const dir = this.cam.dir;
          this.yaw = Math.atan2(dir.x, dir.z);
          this.pitch = Math.asin(dir.y);
        } else if (oldMode === 'firstPerson') {
          this.cam.setOrientation();
          this.targetAzimuth = this.cam.azimuth;
          this.targetElevator = this.cam.elevator;
          this.targetDist = this.cam.dist;
          this.targetAt = this.cam.at;
        }
      }
    });
  } /** End of 'constructor' function */

  /**
   * Response control function.
   * @param none.
   */
  public async response(): Promise<void> {
    input.response();
    
    const isShift = input.isKeyPressed("ShiftLeft") || input.isKeyPressed("ShiftRight");
    const isCtrl = input.isKeyPressed("ControlLeft") || input.isKeyPressed("ControlRight") || 1;
    
    if (input.isKeyJustPressed("Space"))
      timer.isPause = !timer.isPause;

    if (this.cameraMode === 'firstPerson') {
      if (input.leftClick) {
        this.yaw -= input.mouseDX * 0.002 * this.rotateSpeed;
        this.pitch -= input.mouseDY * 0.002 * this.rotateSpeed;
        this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch));
      }
      
      const forward = new vec3(
        Math.sin(this.yaw) * Math.cos(this.pitch),
        Math.sin(this.pitch),
        Math.cos(this.yaw) * Math.cos(this.pitch)
      );
      const right = new vec3(Math.sin(this.yaw - Math.PI / 2), 0, Math.cos(this.yaw - Math.PI / 2));
      
      const speed = this.moveSpeed * 0.1 * (isShift ? 3 : 1);
      let movement = new vec3(0, 0, 0);
      
      if (input.isKeyPressed("KeyW")) movement = movement.add(forward.mulNum(speed));
      if (input.isKeyPressed("KeyS")) movement = movement.add(forward.mulNum(-speed));
      if (input.isKeyPressed("KeyA")) movement = movement.add(right.mulNum(-speed));
      if (input.isKeyPressed("KeyD")) movement = movement.add(right.mulNum(speed));
      
      const newPos = this.cam.loc.add(movement);
      const newAt = newPos.add(forward);
      this.cam.set(newPos, newAt, new vec3(0, -1, 0));
      
      if ((movement.length() > 0.001 || input.leftClick) && this.onCameraMove) {
        this.onCameraMove();
      }
      return;
    }

    const hasMouseMovement = input.mouseDX !== 0 || input.mouseDY !== 0 || input.mouseDZ !== 0;
    const hasArrowInput = input.isKeyPressed("ArrowLeft") || input.isKeyPressed("ArrowRight") || 
                         input.isKeyPressed("ArrowUp") || input.isKeyPressed("ArrowDown");
    const hasDistanceInput = input.isKeyPressed("Minus") || input.isKeyPressed("Equal");
    
    // Third person camera with smooth interpolation
    if (isCtrl && (hasMouseMovement || hasArrowInput || hasDistanceInput || input.leftClick || input.rightClick)) {
      // Ensure target values are initialized
      if (isNaN(this.targetAzimuth)) {
        this.cam.setOrientation();
        this.targetAzimuth = this.cam.azimuth;
        this.targetElevator = this.cam.elevator;
        this.targetDist = this.cam.dist;
        this.targetAt = this.cam.at;
      }
      
      // Update target values based on input
      this.targetAzimuth += 
        (-(input.leftClick ? 1 : 0) *
          timer.globalDeltaTime *
          3.0 *
          (3.5 * input.mouseDX) +
        ((input.isKeyPressed("ArrowLeft") ? 1 : 0) * 0.5 +
          (input.isKeyPressed("ArrowRight") ? 1 : 0) * -0.5)) *
          (1 + Number(isShift) * 3) * this.rotateSpeed;

      this.targetElevator += 
        ((input.leftClick ? 1 : 0) *
          timer.globalDeltaTime *
          2.5 *
          (3.5 * input.mouseDY) +
        ((input.isKeyPressed("ArrowUp") ? 1 : 0) * -0.5 + (input.isKeyPressed("ArrowDown") ? 1 : 0) * 0.5)) *
          (1 + Number(isShift) * 3) * this.rotateSpeed;

      if (this.targetElevator < 0.08) this.targetElevator = 0.08;
      else if (this.targetElevator > 178.9) this.targetElevator = 178.9;

      this.targetDist += 
        -(0.01 * input.mouseDZ) * (1 + Number(isShift) * 4) +
        ((input.isKeyPressed("Minus") ? 1 : 0) * 0.15 + (input.isKeyPressed("Equal") ? 1 : 0) * -0.15) *
          (1 + Number(isShift) * 3);
      if (this.targetDist < 0.1) this.targetDist = 0.1;
      
      if (input.rightClick) {
        // Update target at position for panning
        const panSpeed = (0.3 + Number(isShift) * 0.2) * this.moveSpeed;
        const rightVec = this.cam.right.mulNum(-input.mouseDX * panSpeed * 0.005);
        const upVec = this.cam.up.mulNum(input.mouseDY * panSpeed * 0.005);
        this.targetAt = this.targetAt.add(rightVec).add(upVec);
      }
    }

    // Always interpolate towards target values for smooth movement
    if (!isNaN(this.targetAzimuth)) {
      const deltaTime = timer.globalDeltaTime;
      const lerpFactor = Math.min(1.0, this.smoothFactor * deltaTime * 60);
      
      // Smooth interpolation
      this.cam.azimuth += (this.targetAzimuth - this.cam.azimuth) * lerpFactor;
      this.cam.elevator += (this.targetElevator - this.cam.elevator) * lerpFactor;
      this.cam.dist += (this.targetDist - this.cam.dist) * lerpFactor;
      this.cam.at = this.cam.at.add(this.targetAt.sub(this.cam.at).mulNum(lerpFactor));
      
      // Update camera position
      const elevatorRad = this.cam.elevator * Math.PI / 180;
      const azimuthRad = this.cam.azimuth * Math.PI / 180;
      
      const cameraPos = new vec3(
        this.cam.at.x + this.cam.dist * Math.sin(elevatorRad) * Math.cos(azimuthRad),
        this.cam.at.y + this.cam.dist * Math.cos(elevatorRad),
        this.cam.at.z + this.cam.dist * Math.sin(elevatorRad) * Math.sin(azimuthRad)
      );
      
      this.cam.set(cameraPos, this.cam.at, new vec3(0, -1, 0));
      
      // Check for significant movement AFTER interpolation
      const posDiff = this.cam.loc.sub(this.lastCameraPos).length();
      const atDiff = this.cam.at.sub(this.lastCameraAt).length();
      if ((posDiff > this.movementThreshold || atDiff > this.movementThreshold) && this.onCameraMove) {
        this.onCameraMove();
        this.lastCameraPos = this.cam.loc;
        this.lastCameraAt = this.cam.at;
      }
    }
  } /** End of 'response' function */

}

/** EXPORTS */
export { control }

/** END OF 'control.ts' FILE */
