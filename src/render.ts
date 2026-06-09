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
import { buffer } from "./wgpu/buffers"
import { vec3 } from "./mth/mth_vec3"
import { vec4 } from "./mth/mth_vec4"
import { mat4 } from "./mth/mth_matr"
import { control } from "./input/control"
import { timer } from "./input/timer"
import { input } from "./input/input"

/** 
 * * Primitive class 
 **/
class primitive
{
  public scale2 : vec3;
  public position : vec3;
  public scale : vec3;
  public rotation : vec4;
  public opacity : number;
  public color : vec4;
  public index: number;

  /**
   * @info Primitive constructor
   * @param pos : position vector
   * @param scl : scale vector
   * @param rot : quaternion rotation
   * @param opac : opacity value
   * @param col : color vector
   */
  public constructor(pos : vec3, scl : vec3, rot : vec4, opac : number, col : vec4)
  {
    this.position = pos;
    this.scale = scl;
    this.rotation = rot;
    this.opacity = opac;
    this.color = col;
    this.scale2 = new vec3(scl.x * scl.x, scl.y * scl.y, scl.z * scl.z); 
  } /** End of 'primitive' constructor */

  /**
   * @info Compute covariance matrix
   * @param None.
   * @returns covariance matrix.
   */
  public computeCovariance(): mat4
  {
    let s = mat4.scale(this.scale2);
    let r = this.rotation.q2m();
    let rt = r.transpose();

    return r.multiply(s.multiply(rt));
  } /** End of 'computeCovariance' function */
} /** End of 'primitive' class */

/** 
 * * Render class 
 **/
class render extends core
{
  private controls: control | undefined = undefined;
  private commandEncoder: GPUCommandEncoder | null = null;
  private passEncoder: GPURenderPassEncoder | null = null;
  private computePassEncoder: GPUComputePassEncoder | null = null;
  private renderTexture: GPUTexture | null = null;
  private renderTextureView: GPUTextureView | null = null;

  private workGroupSize = 64;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private inputBuffer: buffer | null = null;
  private outputBuffer: buffer | null = null;
  private cameraBuffer: buffer | null = null;
  private cameraData: Float32Array | null = null;

  private primitives: primitive[] = [];

  /**
   * @info Initialize context function
   * @param Canvas elemant id
   * @returns None.
   */
  public async init(id: Element) {
    await this.coreInit(id);
    this.controls = new control();

    this.renderTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height], 
      format: 'rgba8unorm', 
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
    });
    this.renderTextureView = this.renderTexture.createView();
    
    this.inputBuffer = new buffer(this);
    this.outputBuffer = new buffer(this);
    this.cameraBuffer = new buffer(this);

    this.inputBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: 1024,
    });

    this.outputBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      size: 1024,
    });
    this.cameraBuffer.create({
      size: 4 * (64 + 64 + 64 + 16 * 5),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      type: "uniform",
    });

    const shaderModule = await this.loadShaderModule("src/shaders/compute/comp.wgsl");
    
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' }
    });

    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { 
          binding: 0,
          resource: { buffer: this.cameraBuffer.buffer }
        },
        { 
          binding: 1, 
          resource: { buffer: this.inputBuffer.buffer } 
        },
        { 
          binding: 2, 
          resource: { buffer: this.outputBuffer.buffer } 
        }
      ]
    });
  } /** End of 'constructor' function */

  /**
   * @info Load shader module function
   * @param url : shader url address
   * @returns None.
   */
  private async loadShaderModule(url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch shader at ${url}. Status: ${response.status}`);
    }
    let shaderCode = await response.text();

    const shaderModule = this.device.createShaderModule({
      label: url,
      code: shaderCode
    });

    const compilationInfo = await shaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
        let hasErrors = false;
        console.group(`WGSL Compilation Info for: ${url}`);
        for (const msg of compilationInfo.messages) {
            const logLine = `${msg.type.toUpperCase()} [Line ${msg.lineNum}, Pos ${msg.linePos}]: ${msg.message}`;
            if (msg.type === 'error') {
                console.error(logLine);
                hasErrors = true;
            } else if (msg.type === 'warning') {
                console.warn(logLine);
            } else {
                console.log(logLine);
            }
        }
        console.groupEnd();
        if (hasErrors) throw new Error(`WGSL Compilation failed for ${url}`);
    }

    return shaderModule;
  } /** End of 'loadShaderModule' function */

  /**
   * @info Start render function
   * @param None. 
   * @returns None.
   */
  public start() {
    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(), 
        clearValue: { r: 0.8, g: 0.5, b: 0.5, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
    };

    this.commandEncoder = this.device.createCommandEncoder();
    //this.passEncoder = this.commandEncoder.beginRenderPass(renderPassDescriptor);
    this.controls.response();
    timer.response();
  } /** End of 'start' function */

  /**
   * @info Draw primitive function
   * @param primitive : primitive to draw
   * @returns None.
   */
  public draw(prim: primitive) {


  } /** End of 'draw' function */

  /**
   * @info Draw low level function
   * @returns None.
   */
  private drawLow() {
    if (this.primitives.length == 0) 
      return;
    
    if (!this.cameraData) 
      this.cameraData = new Float32Array(64 + 64 + 64 + 16 * 5);

    let offset = 0;
    this.cameraData.set(this.controls.cam.view.m.flat(), offset); offset += 16;
    this.cameraData.set(this.controls.cam.proj.m.flat(), offset); offset += 16;
    this.cameraData.set(this.controls.cam.vp.m.flat(), offset); offset += 16;
    this.cameraData.set([this.controls.cam.loc.x, this.controls.cam.loc.y, this.controls.cam.loc.z, this.controls.cam.frameW], offset); offset += 4;
    this.cameraData.set([this.controls.cam.at.x, this.controls.cam.at.y, this.controls.cam.at.z, this.controls.cam.frameH], offset); offset += 4;
    this.cameraData.set([this.controls.cam.dir.x, this.controls.cam.dir.y, this.controls.cam.dir.z, this.controls.cam.projDist], offset); offset += 4;
    this.cameraData.set([this.controls.cam.right.x, this.controls.cam.right.y, this.controls.cam.right.z, this.controls.cam.wp], offset); offset += 4;
    this.cameraData.set([this.controls.cam.up.x, this.controls.cam.up.y, this.controls.cam.up.z, this.controls.cam.hp], offset);
      
    this.cameraBuffer.update(this.cameraData);  

    // Collect data
    const gaussians = this.primitives.map(p => ({
      position: p.position,
      scale: p.scale,
      rotation: p.rotation,
      color: p.color,
      opacity: p.opacity,
      index: p.index
    }));
    
    const ELEMENTS_PER_STRUCT = 16; 
    const totalBufferSize = gaussians.length * ELEMENTS_PER_STRUCT * 4; 
    
    const arrayBuffer = new ArrayBuffer(totalBufferSize);
    const floatView = new Float32Array(arrayBuffer);
    const uintView = new Uint32Array(arrayBuffer);
    
    for (let i = 0; i < gaussians.length; i++) {
      const g = gaussians[i];
      const offset = i * ELEMENTS_PER_STRUCT; 
      
      floatView[offset + 0] = g.position.x;
      floatView[offset + 1] = g.position.y;
      floatView[offset + 2] = g.position.z;
      
      floatView[offset + 3] = g.scale.x;
      
      floatView[offset + 4] = g.rotation.x;
      floatView[offset + 5] = g.rotation.y;
      floatView[offset + 6] = g.rotation.z;
      floatView[offset + 7] = g.rotation.w;
      floatView[offset + 8] = g.color.x;
      floatView[offset + 9] = g.color.y;
      floatView[offset + 10] = g.color.z;
      floatView[offset + 11] = g.color.w;
      
      floatView[offset + 12] = g.opacity;
      
      uintView[offset + 13] = g.index;
    }
    const workgroupCount = gaussians.length / this.workGroupSize;
    
    this.inputBuffer.updateArray(arrayBuffer);
    
    if (this.inputBuffer.isSizeChanged)
    {
      this.inputBuffer.isSizeChanged = false;

      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { 
            binding: 0,
            resource: { buffer: this.cameraBuffer.buffer }
          },
          { 
            binding: 1, 
            resource: { buffer: this.inputBuffer.buffer } 
          },
          { 
            binding: 2, 
            resource: { buffer: this.outputBuffer.buffer } 
          }
        ]
      });
    }
      
    this.computePassEncoder = this.commandEncoder.beginComputePass();
    this.computePassEncoder.setPipeline(this.pipeline);
    this.computePassEncoder.setBindGroup(0, this.bindGroup);
    this.computePassEncoder.dispatchWorkgroups(workgroupCount, 1, 1);
    this.computePassEncoder.end();
  } /** End of 'drawLow' function */

  /**
   * @info End render function
   * @param None. 
   * @returns None.
   */
  public end() {
    if (!this.commandEncoder) return;

    this.drawLow();
    
    this.queue.submit([this.commandEncoder.finish()]);
    
    this.passEncoder = null;
    this.commandEncoder = null;
    input.clear();
  } /** End of 'end' function */
} /** End of 'render' class */

export { render }; 

/** END OF 'render.ts' FILE */