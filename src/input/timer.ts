/***************************************************************
 * Copyright (C) 2026
 *    Computer Graphics Support Group of 30 Phys-Math Lyceum
 ***************************************************************/

/** FILE NAME   : timer.ts
 *  PURPOSE     : Gaussian splatting project.
 *  PROGRAMMER  : CGSG'An'2026.
 *                Timofey Hudyakov (TH4).
 *  LAST UPDATE : 09.06.2026
 * 
 *  No part of this file may be changed without agreement of
 *  Computer Graphics Support Group of 30 Phys-Math Lyceum
 */

/** 
 * * Timer class 
 **/
class _timer {
  /** #private parameters */
  private startTime: number = 0;
  private oldTime: number = 0;
  private oldTimeFPS: number = 0;
  private pauseTime: number = 0;
  private frameCounter: number = 0;

  /**
   * @info Get time function
   * @returns current time in seconds
   */
  private getTime = () => {
    return performance.now() / 1000.0;
  }; /** End of 'getTime' function */

  /** #public parameters */
  public globalTime: number = 0;
  public globalDeltaTime: number = 0;
  public time: number = 0;
  public deltaTime: number = 0;
  public FPS: number = 30;
  public isPause: boolean = false;

  /**
   * @info Initialize timer function
   * @returns none
   */
  public initTimer = () => {
    this.globalTime = this.time = this.getTime();
    this.startTime = this.oldTime = this.oldTimeFPS = this.globalTime;
  }; /** End of 'initTimer' function */

  /**
   * @info Timer response function
   * @returns none
   */
  public response() {
    let t = this.getTime();
    // Global time
    this.globalTime = t;
    this.globalDeltaTime = t - this.oldTime;
    // Time with pause
    if (this.isPause) {
      this.deltaTime = 0;
      this.pauseTime += t - this.oldTime;
    } else {
      this.deltaTime = this.globalDeltaTime;
      this.time = t - this.pauseTime - this.startTime;
    }
    // FPS
    this.frameCounter++;
    if (t - this.oldTimeFPS > 1) {
      this.FPS = this.frameCounter / (t - this.oldTimeFPS);
      this.oldTimeFPS = t;
      this.frameCounter = 0;
      this.FPS = Number(this.FPS.toFixed(3));
    }
    this.oldTime = t;
  } /** End of 'response' function */
} /** End of 'timer' class */

/** Timer variable */
const timer: _timer = new _timer();

/** Export to window for UI access */
(window as any).timer = timer;

/** EXPORTS */
export { timer };

/** END OF 'timer.ts' FILE */

