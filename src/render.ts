/***************************************************************
 * Copyright (C) 2026
 *    Computer Graphics Support Group of 30 Phys-Math Lyceum
 ***************************************************************/

/** FILE NAME   : render.ts
 *  PURPOSE     : Gaussian splatting project.
 *  PROGRAMMER  : CGSG'An'2026.
 *                Timofey Hudyakov (TH4).
 *  LAST UPDATE : 08.06.2026
 * 
 *  No part of this file may be changed without agreement of
 *  Computer Graphics Support Group of 30 Phys-Math Lyceum
 */

import { core } from "./wgpu/core";

class render extends core
{
  private renderTexture: GPUTexture | null = null;
  private commandEncoder: GPUCommandEncoder | null = null;
  private passEncoder: GPURenderPassEncoder | null = null;

  /**
   * @info Initialize context function
   * @param Canvas elemant id
   */
  public async init(id: Element) {
    await this.coreInit(id);

    this.renderTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height], 
      format: navigator.gpu.getPreferredCanvasFormat(), 
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
    });
  } /** End of 'constructor' function */

  /**
   * @info Start render function
   * @param None. 
   */
  public async start() {
    const renderPassDescriptor: GPURenderPassDescriptor = {
     colorAttachments: [
       {
         view: this.context.getCurrentTexture().createView(), 
         clearValue: [0.98, 0.68, 0.85, 1.0],
         loadOp: "clear",
         storeOp: "store",
        },
      ],
    };

    this.commandEncoder = this.device.createCommandEncoder();
    this.passEncoder = this.commandEncoder.beginRenderPass(renderPassDescriptor);
  } /** End of 'start' function */

  /**
   * @info End render function
   * @param None. 
   */
  public async end() {
    if (!this.passEncoder || !this.commandEncoder) return;

    this.passEncoder.end();

    this.queue.submit([this.commandEncoder.finish()]);

    this.passEncoder = null;
    this.commandEncoder = null;
  } /** End of 'end' function */
} /** End of 'render' class */

export { render }; 

/** END OF 'render.ts' FILE */