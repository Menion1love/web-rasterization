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
import { control } from "./input/control"
import { timer } from "./input/timer"
import { input } from "./input/input"
import { RadixSortKernel } from 'webgpu-radix-sort';

/** 
 * * Primitive class 
 **/
export class primitive
{
  public position : vec3;
  public scale : vec3;
  public rotation : vec4;
  public opacity : number;
  public color : vec4;
  public index: number = 0;

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
  } /** End of 'primitive' constructor */
} /** End of 'primitive' class */

/** 
 * * Render class 
 **/
class render extends core
{
  private controls!: control;
  private commandEncoder!: GPUCommandEncoder | null;
  private passEncoder!: GPURenderPassEncoder | null;
  private computePassEncoder!: GPUComputePassEncoder | null;
  private renderTexture!: GPUTexture | null;
  private renderTextureView!: GPUTextureView | null;

  private radixSortKernel!: RadixSortKernel;
  private workGroupSize = 64;
  private computePipeline!: GPUComputePipeline;
  private keysPipeline!: GPUComputePipeline;
  private tilePipeline!: GPUComputePipeline;
  private rasterPipeline!: GPUComputePipeline;
  private displayPipeline!: GPURenderPipeline;
  private bindGroup!: GPUBindGroup;
  private tileBindGroup!: GPUBindGroup;
  private keysBindGroup!: GPUBindGroup;
  private rasterBindGroup!: GPUBindGroup;
  private displayBindGroup!: GPUBindGroup;
  private inputBuffer!: buffer;
  private outputBuffer!: buffer;
  private keysBuffer!: buffer;
  private valuesBuffer!: buffer;
  private tileBuffer!: buffer;
  private counterBuffer!: buffer;
  private stagingBuffer!: buffer;
  private cameraBuffer!: buffer;
  private cameraData!: Float32Array;

  private primitives: primitive[] = [];
  private globalKeysCount: number = 1;

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
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING  
    });
    this.renderTextureView = this.renderTexture.createView();
    
    this.inputBuffer = new buffer(this);
    this.outputBuffer = new buffer(this);
    this.cameraBuffer = new buffer(this);
    this.keysBuffer = new buffer(this);
    this.counterBuffer = new buffer(this);
    this.valuesBuffer = new buffer(this);
    this.tileBuffer = new buffer(this);
    this.stagingBuffer = new buffer(this);

    this.stagingBuffer.create({
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      size: 4,
    })

    this.inputBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: 64,
      label: "input",
    });

    this.outputBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      size: 64,
      label: "output",
    });

    this.cameraBuffer.create({
      size: (64 + 64 + 64 + 16 * 5) * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      type: "uniform",
      label: "camera",
    });
    
    let tileSizex = Math.ceil(this.canvas.width / 16);
    let tileSizey = Math.ceil(this.canvas.height / 16);
    
    this.keysBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: 4,
      label: "keys",
    })

    this.valuesBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: 4,
      label: "values",
    })

    this.tileBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: tileSizex * tileSizey * 8,
      label: "tile",
    })

    this.counterBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      size: 4,
    })

    const shaderModule = await this.loadShaderModule("src/shaders/compute/comp.wgsl");
    
    this.computePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' }
    });

    this.bindGroup = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
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
        },
        { 
          binding: 3, 
          resource: { buffer: this.counterBuffer.buffer } 
        }
      ]
    });

    const tileShaderModule = await this.loadShaderModule("src/shaders/compute/tailer.wgsl");

    this.tilePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: tileShaderModule, entryPoint: 'main' }
    });

    this.tileBindGroup = this.device.createBindGroup({
      layout: this.tilePipeline.getBindGroupLayout(0),
      entries: [
        { 
          binding: 0, 
          resource: { buffer: this.keysBuffer.buffer } 
        },
        { 
          binding: 1, 
          resource: { buffer: this.tileBuffer.buffer }
        },
      ]
    });

    const keysShaderModule = await this.loadShaderModule("src/shaders/compute/keys.wgsl");

    this.keysPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: keysShaderModule, entryPoint: 'main' }
    });

    this.keysBindGroup = this.device.createBindGroup({
      layout: this.keysPipeline.getBindGroupLayout(0),
      entries: [
        { 
          binding: 0,
          resource: { buffer: this.cameraBuffer.buffer }
        },
        { 
          binding: 1, 
          resource: { buffer: this.outputBuffer.buffer } 
        },
        { 
          binding: 2, 
          resource: { buffer: this.keysBuffer.buffer } 
        },
        { 
          binding: 3, 
          resource: { buffer: this.valuesBuffer.buffer } 
        },
        { 
          binding: 4, 
          resource: { buffer: this.counterBuffer.buffer } 
        }
      ]
    });

    const rasterShaderModule = await this.loadShaderModule("src/shaders/compute/raster.wgsl");

    this.rasterPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: rasterShaderModule, entryPoint: 'main' }
    });

    this.rasterBindGroup = this.device.createBindGroup({
      layout: this.rasterPipeline.getBindGroupLayout(0),
      entries: [
        { 
          binding: 0,
          resource: { buffer: this.cameraBuffer.buffer }
        },
        { 
          binding: 1, 
          resource: { buffer: this.outputBuffer.buffer } 
        },
        { 
          binding: 2, 
          resource: { buffer: this.valuesBuffer.buffer } 
        },
        { 
          binding: 3, 
          resource: { buffer: this.tileBuffer.buffer } 
        },
        { 
          binding: 4, 
          resource: this.renderTextureView 
        },
      ]
    });

    const displayShaderModule = await this.loadShaderModule("src/shaders/display/display.wgsl");

    const sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });

    this.displayPipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: displayShaderModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: displayShaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.context.getCurrentTexture().format,
        }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });

    this.displayBindGroup = this.device.createBindGroup({
      layout: this.displayPipeline.getBindGroupLayout(0),
      entries: [
        { 
          binding: 0,
          resource: this.renderTextureView
        },
        {
          binding: 1,
          resource: sampler
        }
      ]
    });

    this.radixSortKernel = new RadixSortKernel({
      device: this.device,
      keys: this.keysBuffer.buffer,
      values: this.valuesBuffer.buffer,
      count: 1,
      check_order: false,
      bit_count: 32,
      workgroup_size: { x: 16, y: 16 },
    })

  } /** End of 'constructor' function */

  /**
   * @info Load shader module function
   * @param url : shader url address
   * @returns None.
   */
  private async loadShaderModule(url: string) {
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
    this.commandEncoder = this.device.createCommandEncoder();
    this.controls.response();
    timer.response();
  } /** End of 'start' function */

  /**
   * @info Draw primitive function
   * @param primitive : primitive to draw
   * @returns None.
   */
  public draw(prim: primitive) {
    this.primitives.push(prim);
  } /** End of 'draw' function */

  private async readCounter() {
    await this.stagingBuffer.buffer.mapAsync(GPUMapMode.READ);

    const arrayBuffer = this.stagingBuffer.buffer.getMappedRange();

    const view = new Uint32Array(arrayBuffer);
    
    const globalKeysCount = view[0];

    this.stagingBuffer.buffer.unmap();

    return globalKeysCount;
  }

  /**
   * @info Draw low level function
   * @returns None.
   */
  private async drawLow() {
    if (this.primitives.length == 0 || !this.commandEncoder) 
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
      
    await this.cameraBuffer.update(this.cameraData);
    await this.counterBuffer.updateInteger(new Uint32Array([0]));
    await this.keysBuffer.resize(this.globalKeysCount * 4, true);
    await this.valuesBuffer.resize(this.globalKeysCount * 4, true);

    this.commandEncoder.clearBuffer(this.tileBuffer.buffer);
    this.commandEncoder.clearBuffer(this.valuesBuffer.buffer);
    this.commandEncoder.clearBuffer(this.keysBuffer.buffer);

    // Collect data
    const gaussians = this.primitives.map(p => ({
      position: p.position,
      scale: p.scale,
      rotation: p.rotation,
      color: p.color,
      opacity: p.opacity,
      index: p.index
    }));
    
    const totalBufferSize = gaussians.length * 16 * 4; 
    
    const arrayBuffer = new ArrayBuffer(totalBufferSize);
    const floatView = new Float32Array(arrayBuffer);
    const uintView = new Uint32Array(arrayBuffer);
    
    for (let i = 0; i < gaussians.length; i++) {
      const g = gaussians[i];
      const offset = i * 16; 
      
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
    await this.inputBuffer.updateArray(arrayBuffer);
    
    let tileSizex = Math.ceil(this.canvas.width / 16);
    let tileSizey = Math.ceil(this.canvas.height / 16);
    
    if (this.inputBuffer.isSizeChanged)
    {
      await this.outputBuffer.resize(gaussians.length * 16 * 4);
      this.inputBuffer.isSizeChanged = false;
      this.outputBuffer.isSizeChanged = false;

      this.bindGroup = this.device.createBindGroup({
        layout: this.computePipeline.getBindGroupLayout(0),
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
          },
          { 
            binding: 3, 
            resource: { buffer: this.counterBuffer.buffer } 
          }
        ]
      });

      
      this.keysBindGroup = this.device.createBindGroup({
        layout: this.keysPipeline.getBindGroupLayout(0),
        entries: [
          { 
            binding: 0,
            resource: { buffer: this.cameraBuffer.buffer }
          },
          { 
            binding: 1, 
            resource: { buffer: this.outputBuffer.buffer } 
          },
          { 
            binding: 2, 
            resource: { buffer: this.keysBuffer.buffer } 
          },
          { 
            binding: 3, 
            resource: { buffer: this.valuesBuffer.buffer } 
          },
          { 
            binding: 4, 
            resource: { buffer: this.counterBuffer.buffer } 
          }
        ]
      });

      this.rasterBindGroup = this.device.createBindGroup({
        layout: this.rasterPipeline.getBindGroupLayout(0),
        entries: [
          { 
            binding: 0,
            resource: { buffer: this.cameraBuffer.buffer }
          },
          { 
            binding: 1, 
            resource: { buffer: this.outputBuffer.buffer } 
          },
          { 
            binding: 2, 
            resource: { buffer: this.valuesBuffer.buffer } 
          },
          { 
            binding: 3, 
            resource: { buffer: this.tileBuffer.buffer } 
          },
          { 
            binding: 4, 
            resource: this.renderTextureView 
          },
        ]
      });
    }

    this.computePassEncoder = this.commandEncoder.beginComputePass({});
    
    // Project points and get matrices
    this.computePassEncoder.setPipeline(this.computePipeline);
    this.computePassEncoder.setBindGroup(0, this.bindGroup);
    this.computePassEncoder.dispatchWorkgroups(Math.floor((gaussians.length + 63) / this.workGroupSize), 1, 1);
    this.computePassEncoder.end();

    this.commandEncoder.copyBufferToBuffer(
      this.counterBuffer.buffer,
      0,
      this.stagingBuffer.buffer,
      0,
      4
    );

    // Resize keys and values buffer
    this.globalKeysCount = await this.readCounter();
    console.log(this.globalKeysCount);

    const resize = async () => {
      if (this.keysBuffer.isSizeChanged || this.valuesBuffer.isSizeChanged)
      {
        this.keysBuffer.isSizeChanged = false;
        this.valuesBuffer.isSizeChanged = false;

        this.tileBindGroup = this.device.createBindGroup({
          layout: this.tilePipeline.getBindGroupLayout(0),
          entries: [
            { 
              binding: 0, 
              resource: { buffer: this.keysBuffer.buffer } 
            },
            { 
              binding: 1, 
              resource: { buffer: this.tileBuffer.buffer }
            },
          ]
        });
        
        this.keysBindGroup = this.device.createBindGroup({
          layout: this.keysPipeline.getBindGroupLayout(0),
          entries: [
            { 
              binding: 0,
              resource: { buffer: this.cameraBuffer.buffer }
            },
            { 
              binding: 1, 
              resource: { buffer: this.outputBuffer.buffer } 
            },
            { 
              binding: 2, 
              resource: { buffer: this.keysBuffer.buffer } 
            },
            { 
              binding: 3, 
              resource: { buffer: this.valuesBuffer.buffer } 
            },
            { 
              binding: 4, 
              resource: { buffer: this.counterBuffer.buffer } 
            }
          ]
        });

        this.rasterBindGroup = this.device.createBindGroup({
          layout: this.rasterPipeline.getBindGroupLayout(0),
          entries: [
            { 
              binding: 0,
              resource: { buffer: this.cameraBuffer.buffer }
            },
            { 
              binding: 1, 
              resource: { buffer: this.outputBuffer.buffer } 
            },
            { 
              binding: 2, 
              resource: { buffer: this.valuesBuffer.buffer } 
            },
            { 
              binding: 3, 
              resource: { buffer: this.tileBuffer.buffer } 
            },
            { 
              binding: 4, 
              resource: this.renderTextureView 
            },
          ]
        });

        this.radixSortKernel = new RadixSortKernel({
          device: this.device,
          keys: this.keysBuffer.buffer,
          values: this.valuesBuffer.buffer,
          count: this.globalKeysCount,
          check_order: false,
          bit_count: 32,
          workgroup_size: { x: 16, y: 16 },
        })
      }
    }
    await resize();

    // Update counter
    this.commandEncoder.clearBuffer(this.counterBuffer.buffer);
    if (this.globalKeysCount == 0) return;

    this.computePassEncoder = this.commandEncoder.beginComputePass({});

    // Fill keys and values
    this.computePassEncoder.setPipeline(this.keysPipeline);
    this.computePassEncoder.setBindGroup(0, this.keysBindGroup);
    this.computePassEncoder.dispatchWorkgroups(Math.floor((gaussians.length + 63) / this.workGroupSize), 1, 1);

    // Sort keys
    this.radixSortKernel.dispatch(this.computePassEncoder);

    // Get tile ranges
    this.computePassEncoder.setPipeline(this.tilePipeline);
    this.computePassEncoder.setBindGroup(0, this.tileBindGroup);
    this.computePassEncoder.dispatchWorkgroups(Math.floor((this.globalKeysCount + 255) / 256), 1, 1);

    // Rasterization
    this.computePassEncoder.setPipeline(this.rasterPipeline);
    this.computePassEncoder.setBindGroup(0, this.rasterBindGroup);
    this.computePassEncoder.dispatchWorkgroups(tileSizex, tileSizey, 1);
    this.computePassEncoder.end();
  } /** End of 'drawLow' function */

  /**
   * @info End render function
   * @param None. 
   * @returns None.
   */
  public async end() {
    if (!this.commandEncoder) return;

    const clearPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: this.renderTextureView, 
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
    };
    const clearPass = this.commandEncoder.beginRenderPass(clearPassDescriptor);
    clearPass.end();

    await this.drawLow();

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(), 
        clearValue: { r: 0.8, g: 0.5, b: 0.5, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
    };

    this.passEncoder = this.commandEncoder.beginRenderPass(renderPassDescriptor);

    this.passEncoder.setPipeline(this.displayPipeline);
    this.passEncoder.setBindGroup(0, this.displayBindGroup);
    this.passEncoder.draw(3);
    this.passEncoder.end();

    this.queue.submit([this.commandEncoder.finish()]);
    
    this.passEncoder = null;
    this.commandEncoder = null;
    input.clear();
    this.primitives = [];
  } /** End of 'end' function */
} /** End of 'render' class */

export { render }; 

/** END OF 'render.ts' FILE */