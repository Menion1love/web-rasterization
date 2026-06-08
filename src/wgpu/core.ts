/***************************************************************
 * Copyright (C) 2026
 *    Computer Graphics Support Group of 30 Phys-Math Lyceum
 ***************************************************************/

/** FILE NAME   : core.ts
 *  PURPOSE     : Gaussian splatting project.
 *  PROGRAMMER  : CGSG'An'2026.
 *                Timofey Hudyakov (TH4).
 *  LAST UPDATE : 08.06.2026
 * 
 *  No part of this file may be changed without agreement of
 *  Computer Graphics Support Group of 30 Phys-Math Lyceum
 */

class core 
{
  /** #public parameters */
  public adapter!: GPUAdapter; // Adapter
  public device!: GPUDevice; // Device
  public queue!: GPUQueue; // Queue
  public context!: GPUCanvasContext; // Canvas context
  public canvas!: HTMLCanvasElement; // Canvas element
  public isinit!: boolean;

  /**
   * @info Initialize context function
   * @param Canvas elemant id
   */
  public async coreInit(id: Element) {
    this.canvas = id as HTMLCanvasElement;

    if (!navigator.gpu) {
      throw Error("Web gpu not supported");
    }

    // get adapter
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw Error("No GPU adapter found");
    }
    this.adapter = adapter;

    // get device
    this.device = await this.adapter.requestDevice();

    // get queue
    this.queue = this.device.queue;

    // Set canvas resolution for high DPI displays
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = this.canvas.clientWidth * dpr;
    this.canvas.height = this.canvas.clientHeight * dpr;

    // Create context
    const context = this.canvas.getContext("webgpu");

    if (!context) {
      throw new Error("Failed to get WebGPU context");
    }
    this.context = context as GPUCanvasContext;
    
    this.context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST, 

    });
    this.isinit = true;

    console.log(`WebGPU initialized! Canvas size: ${this.canvas.width}x${this.canvas.height}`);
  } /** End of 'init' function */
} /** End of 'core' class */

/** EXPORTS */
export { core };

/** END OF 'core.ts' FILE */