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
  public controls!: control;
  private commandEncoder!: GPUCommandEncoder | null;
  private passEncoder!: GPURenderPassEncoder | null;
  private computePassEncoder!: GPUComputePassEncoder | null;
  public renderTexture!: GPUTexture | null;
  public renderTextureView!: GPUTextureView | null;

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

  private transmittanceTexture!: GPUTexture | null;
  private transmittanceTextureView!: GPUTextureView | null;
  private gradientTexture!: GPUTexture | null;
  private gradientTextureView!: GPUTextureView | null;
  public imageTexture!: GPUTexture | null;
  public imageTextureView!: GPUTextureView | null;
  private gradientPipeline!: GPUComputePipeline;
  private gradientBindGroup!: GPUBindGroup;
  private backwardPipeline!: GPUComputePipeline;
  private backwardBindGroup!: GPUBindGroup;
  private gGradBuffer!: buffer;
  private learnPipeline!: GPUComputePipeline;
  private learnBindGroup!: GPUBindGroup;
  private adamMBuffer!: buffer;
  private adamVBuffer!: buffer;
  private iteratorBuffer!: buffer;

  private primitives: primitive[] = [];
  private globalKeysCount: number = 1;
  private IterationsCount: number = 1;

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
      format: 'rgba32float', 
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING  
    });
    this.renderTextureView = this.renderTexture.createView();

    this.imageTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height], 
      format: 'rgba32float', 
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING  
    });

    this.imageTextureView = this.imageTexture.createView();
    
    this.gradientTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height], 
      format: 'rgba32float', 
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING  
    });

    this.gradientTextureView = this.gradientTexture.createView();
    
    this.transmittanceTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height], 
      format: 'r32float', 
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING  
    });

    this.transmittanceTextureView = this.transmittanceTexture.createView();
    
    this.inputBuffer = new buffer(this);
    this.outputBuffer = new buffer(this);
    this.cameraBuffer = new buffer(this);
    this.keysBuffer = new buffer(this);
    this.counterBuffer = new buffer(this);
    this.valuesBuffer = new buffer(this);
    this.tileBuffer = new buffer(this);
    this.stagingBuffer = new buffer(this);
    this.gGradBuffer = new buffer(this);
    this.adamMBuffer = new buffer(this);
    this.adamVBuffer = new buffer(this);
    this.iteratorBuffer = new buffer(this);

    this.stagingBuffer.create({
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      size: 4,
    })
    
    this.iteratorBuffer.create({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      type: "uniform",
      label: "iterator buffer",
    });

    this.inputBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: 64,
      label: "input",
    });

    this.adamMBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: 64,
      label: "adam M",
    });

    this.adamVBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: 64,
      label: "adam V",
    });

    this.gGradBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: 48,
      label: "grad",
    })

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

    const shaderModule = await this.loadShaderModule("src/shaders/forward/comp.wgsl");
    
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

    const tileShaderModule = await this.loadShaderModule("src/shaders/forward/tailer.wgsl");

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

    const keysShaderModule = await this.loadShaderModule("src/shaders/forward/keys.wgsl");

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

    const rasterShaderModule = await this.loadShaderModule("src/shaders/forward/raster.wgsl");

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
        { 
          binding: 5, 
          resource: this.transmittanceTextureView 
        },
      ]
    });

    const gradientShaderModule = await this.loadShaderModule("src/shaders/learn/gradient.wgsl");

    this.gradientPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: gradientShaderModule, entryPoint: 'main' }
    });

    this.gradientBindGroup = this.device.createBindGroup({
      layout: this.gradientPipeline.getBindGroupLayout(0),
      entries: [
        { 
          binding: 0, 
          resource: this.renderTextureView 
        },
        { 
          binding: 1, 
          resource: this.imageTextureView 
        },
        { 
          binding: 2, 
          resource: this.gradientTextureView
        },
      ]
    });
    
    const backwardShaderModule = await this.loadShaderModule("src/shaders/learn/backward.wgsl");

    this.backwardPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: backwardShaderModule, entryPoint: 'main' }
    });

    this.backwardBindGroup = this.device.createBindGroup({
      layout: this.backwardPipeline.getBindGroupLayout(0),
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
        { 
          binding: 5, 
          resource: this.transmittanceTextureView 
        },
        { 
          binding: 6, 
          resource: { buffer: this.gGradBuffer.buffer }
        },
      ]
    });

    const learnShaderModule = await this.loadShaderModule("src/shaders/learn/learner.wgsl");

    this.learnPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: learnShaderModule, entryPoint: 'main' }
    });

    this.learnBindGroup = this.device.createBindGroup({
      layout: this.learnPipeline.getBindGroupLayout(0),
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
          resource: { buffer: this.gGradBuffer.buffer } 
        },
        { 
          binding: 3, 
          resource: { buffer: this.adamMBuffer.buffer } 
        },
        { 
          binding: 4, 
          resource: { buffer: this.adamVBuffer.buffer } 
        },
        { 
          binding: 5, 
          resource: { buffer: this.iteratorBuffer.buffer } 
        },
      ]
    });

    const displayShaderModule = await this.loadShaderModule("src/shaders/display/display.wgsl");

    const sampler = this.device.createSampler({
      magFilter: 'nearest', 
      minFilter: 'nearest',  
      mipmapFilter: 'nearest' 
    });

    const displayBindGroupLayout = this.device.createBindGroupLayout({
      label: "Display Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: 'unfilterable-float', 
            viewDimension: '2d',
            multisampled: false
          }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: {
            type: 'non-filtering' 
          }
        }
      ]
    });
    
    this.displayPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [displayBindGroupLayout] 
      }),
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
      layout: displayBindGroupLayout,
      entries: [
        { 
          binding: 0,
          resource: this.gradientTextureView
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

  public reload()
  {
    this.gradientBindGroup = this.device.createBindGroup({
      layout: this.gradientPipeline.getBindGroupLayout(0),
      entries: [
        { 
          binding: 0, 
          resource: this.renderTextureView 
        },
        { 
          binding: 1, 
          resource: this.imageTextureView 
        },
        { 
          binding: 2, 
          resource: this.gradientTextureView
        },
      ]
    });
  }

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

  public async primsClear() {
    this.primitives = [];
  }

  /**
   * @info Attach primitives to draw
   * @param primitive[] : array of primitives to draw
   * @returns None.
   */
  public async attachToDraw(prims: primitive[]) {
    for (let i = 0; i < prims.length; i++)
      this.draw(prims[i]);

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
      floatView[offset + 3] = g.opacity;
      
      floatView[offset + 4] = g.scale.x;
      floatView[offset + 5] = g.scale.y;
      floatView[offset + 6] = g.scale.z;
      uintView[offset + 7] = g.index;
      
      floatView[offset + 8] = g.rotation.x;
      floatView[offset + 9] = g.rotation.y;
      floatView[offset + 10] = g.rotation.z;
      floatView[offset + 11] = g.rotation.w;
      floatView[offset + 12] = g.color.x;
      floatView[offset + 13] = g.color.y;
      floatView[offset + 14] = g.color.z;
      floatView[offset + 15] = g.color.w;
    }
    await this.inputBuffer.updateArray(arrayBuffer);
  }

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
    await this.iteratorBuffer.updateInteger(new Uint32Array([this.IterationsCount]));

    this.commandEncoder.clearBuffer(this.tileBuffer.buffer);
    this.commandEncoder.clearBuffer(this.valuesBuffer.buffer);
    this.commandEncoder.clearBuffer(this.keysBuffer.buffer);
    this.commandEncoder.clearBuffer(this.gGradBuffer.buffer);
    
    let tileSizex = Math.ceil(this.canvas.width / 16);
    let tileSizey = Math.ceil(this.canvas.height / 16);

    const len = this.primitives.length;
    
    if (this.inputBuffer.isSizeChanged)
    {
      await this.outputBuffer.resize(len * 16 * 4);
      await this.gGradBuffer.resize(len * 12 * 4);
      await this.adamMBuffer.resize(len * 16 * 4);
      await this.adamVBuffer.resize(len * 16 * 4);
      this.commandEncoder.clearBuffer(this.adamMBuffer.buffer);
      this.commandEncoder.clearBuffer(this.adamVBuffer.buffer);
      this.commandEncoder.clearBuffer(this.gGradBuffer.buffer);

      this.gGradBuffer.isSizeChanged = false;
      this.adamVBuffer.isSizeChanged = false;
      this.adamMBuffer.isSizeChanged = false;
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
          { 
            binding: 5, 
            resource: this.transmittanceTextureView 
          },
        ]
      });
      
      this.backwardBindGroup = this.device.createBindGroup({
        layout: this.backwardPipeline.getBindGroupLayout(0),
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
          { 
            binding: 5, 
            resource: this.transmittanceTextureView 
          },
          { 
            binding: 6, 
            resource: { buffer: this.gGradBuffer.buffer }
          },
        ]
      });

      this.learnBindGroup = this.device.createBindGroup({
        layout: this.learnPipeline.getBindGroupLayout(0),
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
            resource: { buffer: this.gGradBuffer.buffer } 
          },
          { 
            binding: 3, 
            resource: { buffer: this.adamMBuffer.buffer } 
          },
          { 
            binding: 4, 
            resource: { buffer: this.adamVBuffer.buffer } 
          },
          { 
            binding: 5, 
            resource: { buffer: this.iteratorBuffer.buffer } 
          },
        ]
      });

    }

    this.computePassEncoder = this.commandEncoder.beginComputePass({});
    
    // Project points and get matrices
    this.computePassEncoder.setPipeline(this.computePipeline);
    this.computePassEncoder.setBindGroup(0, this.bindGroup);
    this.computePassEncoder.dispatchWorkgroups(Math.floor((len + 63) / this.workGroupSize), 1, 1);
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
    await this.keysBuffer.resize(this.globalKeysCount * 4, true);
    await this.valuesBuffer.resize(this.globalKeysCount * 4, true);

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
          { 
            binding: 5, 
            resource: this.transmittanceTextureView 
          },
        ]
      });

      this.backwardBindGroup = this.device.createBindGroup({
        layout: this.backwardPipeline.getBindGroupLayout(0),
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
          { 
            binding: 5, 
            resource: this.transmittanceTextureView 
          },
          { 
            binding: 6, 
            resource: { buffer: this.gGradBuffer.buffer }
          },
        ]
      });

      this.radixSortKernel = new RadixSortKernel({
        device: this.device,
        keys: this.keysBuffer.buffer,
        values: this.valuesBuffer.buffer,
        count: this.keysBuffer.bufferDesriptor.size / 4,
        check_order: false,
        bit_count: 32,
        workgroup_size: { x: 16, y: 16 },
      })
    }
    
    // Update counter
    this.commandEncoder.clearBuffer(this.counterBuffer.buffer);
    if (this.globalKeysCount == 0) return;

    this.computePassEncoder = this.commandEncoder.beginComputePass({});

    // Fill keys and values
    this.computePassEncoder.setPipeline(this.keysPipeline);
    this.computePassEncoder.setBindGroup(0, this.keysBindGroup);
    this.computePassEncoder.dispatchWorkgroups(Math.floor((len + 63) / this.workGroupSize), 1, 1);

    // Sort keys
    this.radixSortKernel.dispatch(this.computePassEncoder);

    // Get tile ranges
    // let perWorkgroupSize = Maththis.keysBuffer.bufferDesriptor.size / 12;
    this.computePassEncoder.setPipeline(this.tilePipeline);
    this.computePassEncoder.setBindGroup(0, this.tileBindGroup);
    this.computePassEncoder.dispatchWorkgroups(Math.floor((this.keysBuffer.bufferDesriptor.size / 4 + 255) / 256), 1, 1);

    // Rasterization
    this.computePassEncoder.setPipeline(this.rasterPipeline);
    this.computePassEncoder.setBindGroup(0, this.rasterBindGroup);
    this.computePassEncoder.dispatchWorkgroups(tileSizex, tileSizey, 1);
    this.computePassEncoder.end();

    this.computePassEncoder = this.commandEncoder.beginComputePass({});
    
    // Gradients calculation
    this.computePassEncoder.setPipeline(this.gradientPipeline);
    this.computePassEncoder.setBindGroup(0, this.gradientBindGroup);
    this.computePassEncoder.dispatchWorkgroups(tileSizex, tileSizey, 1);

    // Backward pass
    this.computePassEncoder.setPipeline(this.backwardPipeline);
    this.computePassEncoder.setBindGroup(0, this.backwardBindGroup);
    this.computePassEncoder.dispatchWorkgroups(tileSizex, tileSizey, 1);
    
    // Learn pass 
    this.computePassEncoder.setPipeline(this.learnPipeline);
    this.computePassEncoder.setBindGroup(0, this.learnBindGroup);
    this.computePassEncoder.dispatchWorkgroups(Math.floor((len + 63) / this.workGroupSize), 1, 1);
    this.computePassEncoder.end();
    
    this.IterationsCount++;
  } /** End of 'drawLow' function */

  /**
   * @info End render function
   * @param None. 
   * @returns None.
   */
  public async end(flag: boolean) {
    if (!this.commandEncoder) return;

    if (flag)
    {
      const clearPassDescriptor: GPURenderPassDescriptor = {
        colorAttachments: [
          {
          view: this.renderTextureView, 
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store'
        },
        {
          view: this.gradientTextureView, 
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store'
        }],
      };
      const clearPass = this.commandEncoder.beginRenderPass(clearPassDescriptor);
      clearPass.end();
      await this.drawLow();
    }

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
    // this.primitives = [];
  } /** End of 'end' function */
} /** End of 'render' class */

export { render }; 

/** END OF 'render.ts' FILE */