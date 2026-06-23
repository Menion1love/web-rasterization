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
  private stagingBuffer2!: buffer;
  private cameraBuffer!: buffer;
  private cameraData: Float32Array | undefined;
  private static readonly CAMERA_UNIFORM_FLOATS = 16 * 3 + 4 * 5;

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
  private freeSlotBuffer!: buffer;
  private aliveFlagsBuffer!: buffer;
  private denistyBuffer!: buffer;
  private denistyParamsBuffer!: buffer;
  private denistyclearPipeline!: GPUComputePipeline;
  private denistyclearBindGroup!: GPUBindGroup;
  private denistyctrlPipeline!: GPUComputePipeline;
  private denistyctrlBindGroup!: GPUBindGroup;

  private primitives: primitive[] = [];
  private globalKeysCount: number = 1;
  public IterationsCount: number = 1;
  private warmupIters = 200;
  private densifyInterval = 100;
  private stopDensifyIter = 15000;
  private gradThreshold = 0.003;
  private scaleThreshold = 0.01;
  private opacityThreshold = 0.05;
  private gaussainsCount = 0;
  private FrameID = 0;
  private renderingEnabled = false;

  /**
   * @info Calculate capacity function
   * @param count: count to capacity
   * @returns None.
   */
  public calcCapacity(count: number): number {
    return count * 1;
  } /** End of 'calcCapacity' function */

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
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING  
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
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING  
    });

    this.gradientTextureView = this.gradientTexture.createView();
    
    this.transmittanceTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height], 
      format: 'r32float', 
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING  
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
    this.stagingBuffer2 = new buffer(this);
    this.gGradBuffer = new buffer(this);
    this.adamMBuffer = new buffer(this);
    this.adamVBuffer = new buffer(this);
    this.iteratorBuffer = new buffer(this);
    this.freeSlotBuffer = new buffer(this);
    this.aliveFlagsBuffer = new buffer(this);
    this.denistyBuffer = new buffer(this);
    this.denistyParamsBuffer = new buffer(this);

    this.stagingBuffer.create({
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      size: 4,
    })

    this.stagingBuffer2.create({
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      size: 4,
    })

    this.freeSlotBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      size: 4,
    })

    this.aliveFlagsBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      size: 4,
    })

    this.denistyBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      size: 8,
    })
    
    this.iteratorBuffer.create({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      type: "uniform",
      label: "iterator buffer",
    });

    this.inputBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      size: 64,
      label: "input",
    });

    this.adamMBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      size: 64,
      label: "adam M",
    });

    this.adamVBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      size: 64,
      label: "adam V",
    });

    this.gGradBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      size: 48,
      label: "grad",
    })

    this.outputBuffer.create({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      size: 64,
      label: "output",
    });

    this.cameraBuffer.create({
      size: (64 + 64 + 64 + 16 * 5) * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      type: "uniform",
      label: "camera",
    });

    this.denistyParamsBuffer.create({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      type: "uniform",
      label: "denisty",
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

    const shaderModule = await this.loadShaderModule("shaders/forward/comp.wgsl");
    
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

    const tileShaderModule = await this.loadShaderModule("shaders/forward/tailer.wgsl");

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
        { 
          binding: 2, 
          resource: { buffer: this.counterBuffer.buffer }
        },
      ]
    });

    const keysShaderModule = await this.loadShaderModule("shaders/forward/keys.wgsl");

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

    const rasterShaderModule = await this.loadShaderModule("shaders/forward/raster.wgsl");

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

    const gradientShaderModule = await this.loadShaderModule("shaders/learn/gradient.wgsl");

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
    
    const backwardShaderModule = await this.loadShaderModule("shaders/learn/backward.wgsl");

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
        { 
          binding: 7, 
          resource: { buffer: this.denistyBuffer.buffer }
        },
      ]
    });

    const learnShaderModule = await this.loadShaderModule("shaders/learn/learner.wgsl");

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

    const denClearShaderModule = await this.loadShaderModule("shaders/learn/clear_density.wgsl");

    this.denistyclearPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: denClearShaderModule, entryPoint: 'main' }
    });

    this.denistyclearBindGroup = this.device.createBindGroup({
      layout: this.denistyclearPipeline.getBindGroupLayout(0),
      entries: [
        { 
          binding: 0,
          resource: { buffer: this.denistyBuffer.buffer }
        },
      ]
    });

    const denCtrlShaderModule = await this.loadShaderModule("shaders/learn/density_control.wgsl");

    this.denistyctrlPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: denCtrlShaderModule, entryPoint: 'main' }
    });

    this.denistyctrlBindGroup = this.device.createBindGroup({
      layout: this.denistyctrlPipeline.getBindGroupLayout(0),
      entries: [
        { 
          binding: 0,
          resource: { buffer: this.inputBuffer.buffer }
        },
        { 
          binding: 1, 
          resource: { buffer: this.adamMBuffer.buffer } 
        },
        { 
          binding: 2, 
          resource: { buffer: this.adamVBuffer.buffer } 
        },
        { 
          binding: 3, 
          resource: { buffer: this.aliveFlagsBuffer.buffer } 
        },
        { 
          binding: 4, 
          resource: { buffer: this.denistyBuffer.buffer } 
        },
        { 
          binding: 5, 
          resource: { buffer: this.freeSlotBuffer.buffer } 
        },
        { 
          binding: 6, 
          resource: { buffer: this.denistyParamsBuffer.buffer } 
        },
      ]
    });

    const displayShaderModule = await this.loadShaderModule("shaders/display/display.wgsl");

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
        // { 
        //   binding: 0,
        //   resource: this.gradientTextureView
        // },
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

    this.renderingEnabled = true;
    this.ensureCameraData();
  } /** End of 'constructor' function */

  private ensureCameraData() {
    if (!this.cameraData || this.cameraData.length < render.CAMERA_UNIFORM_FLOATS) {
      this.cameraData = new Float32Array(render.CAMERA_UNIFORM_FLOATS);
    }
  }

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
    if (!this.renderingEnabled) return;
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
    this.gaussainsCount = 0;
  }

  public applyTrainingConfig(config: {
    warmupIters: number;
    densifyInterval: number;
    stopDensifyIter: number;
    gradThreshold: number;
    scaleThreshold: number;
    opacityThreshold: number;
  }) {
    this.warmupIters = config.warmupIters;
    this.densifyInterval = config.densifyInterval;
    this.stopDensifyIter = config.stopDensifyIter;
    this.gradThreshold = config.gradThreshold;
    this.scaleThreshold = config.scaleThreshold;
    this.opacityThreshold = config.opacityThreshold;
  }

  /**
   * @info Replace scene primitives and upload them to GPU buffers
   * @param primitive[] : array of primitives to draw
   * @returns None.
   */
  public async loadScene(prims: primitive[]) {
    this.primitives = [];
    this.gaussainsCount = 0;
    this.IterationsCount = 1;

    for (let i = 0; i < prims.length; i++) {
      this.draw(prims[i]);
    }

    await this.uploadSceneBuffers();
  }

  /**
   * @info Attach primitives to draw
   * @param primitive[] : array of primitives to draw
   * @returns None.
   */
  public async attachToDraw(prims: primitive[]) {
    await this.loadScene(prims);
  }

  private async uploadSceneBuffers() {
    const gaussians = this.primitives.map(p => ({
      position: p.position,
      scale: p.scale,
      rotation: p.rotation,
      color: p.color,
      opacity: p.opacity,
      index: p.index
    }));
    this.gaussainsCount = this.primitives.length;
    const len = this.calcCapacity(this.primitives.length);
    
    const totalBufferSize = len * 16 * 4;
    
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
    await this.outputBuffer.resize(len * 16 * 4);
    await this.gGradBuffer.resize(len * 12 * 4);
    await this.adamMBuffer.resize(len * 16 * 4);
    await this.adamVBuffer.resize(len * 16 * 4);
    await this.aliveFlagsBuffer.resize(len * 4);
    await this.denistyBuffer.resize(len * 2 * 4);

    const aliveArrayBuffer = new ArrayBuffer(len * 4);
    const aliveView = new Uint32Array(aliveArrayBuffer);

    for (let i = 0; i < gaussians.length; i++) {
      aliveView[i] = 1;
    }

    await this.aliveFlagsBuffer.resize(len * 4);
    await this.aliveFlagsBuffer.updateArray(aliveArrayBuffer);

    const densityArrayBuffer = new ArrayBuffer(len * 2 * 4);

    await this.denistyBuffer.resize(len * 2 * 4);
    await this.denistyBuffer.updateArray(densityArrayBuffer);

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
          resource: this.gradientTextureView 
        },
        { 
          binding: 5, 
          resource: this.transmittanceTextureView 
        },
        { 
          binding: 6, 
          resource: { buffer: this.gGradBuffer.buffer }
        },
        { 
          binding: 7, 
          resource: { buffer: this.denistyBuffer.buffer }
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

    this.denistyclearBindGroup = this.device.createBindGroup({
      layout: this.denistyclearPipeline.getBindGroupLayout(0),
      entries: [
        { 
          binding: 0,
          resource: { buffer: this.denistyBuffer.buffer }
        },
      ]
    });
    this.denistyctrlBindGroup = this.device.createBindGroup({
      layout: this.denistyctrlPipeline.getBindGroupLayout(0),
      entries: [
        { 
          binding: 0,
          resource: { buffer: this.inputBuffer.buffer }
        },
        { 
          binding: 1, 
          resource: { buffer: this.adamMBuffer.buffer } 
        },
        { 
          binding: 2, 
          resource: { buffer: this.adamVBuffer.buffer } 
        },
        { 
          binding: 3, 
          resource: { buffer: this.aliveFlagsBuffer.buffer } 
        },
        { 
          binding: 4, 
          resource: { buffer: this.denistyBuffer.buffer } 
        },
        { 
          binding: 5, 
          resource: { buffer: this.freeSlotBuffer.buffer } 
        },
        { 
          binding: 6, 
          resource: { buffer: this.denistyParamsBuffer.buffer } 
        },
      ]
    });
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
  private async drawLow(flag: boolean) {
    if (this.primitives.length == 0 || !this.commandEncoder) 
      return;
    
    this.ensureCameraData();
    const cameraData = this.cameraData!;

    let offset = 0;
    cameraData.set(this.controls.cam.view.m.flat(), offset); offset += 16;
    cameraData.set(this.controls.cam.proj.m.flat(), offset); offset += 16;
    cameraData.set(this.controls.cam.vp.m.flat(), offset); offset += 16;
    cameraData.set([this.controls.cam.loc.x, this.controls.cam.loc.y, this.controls.cam.loc.z, this.controls.cam.frameW], offset); offset += 4;
    cameraData.set([this.controls.cam.at.x, this.controls.cam.at.y, this.controls.cam.at.z, this.controls.cam.frameH], offset); offset += 4;
    cameraData.set([this.controls.cam.dir.x, this.controls.cam.dir.y, this.controls.cam.dir.z, this.controls.cam.projDist], offset); offset += 4;
    cameraData.set([this.controls.cam.right.x, this.controls.cam.right.y, this.controls.cam.right.z, this.controls.cam.wp], offset); offset += 4;
    cameraData.set([this.controls.cam.up.x, this.controls.cam.up.y, this.controls.cam.up.z, this.controls.cam.hp], offset);
      
    const arrayBuffer = new ArrayBuffer(48);
    const floatView = new Float32Array(arrayBuffer);
    const uintView = new Uint32Array(arrayBuffer);
    const caplen = this.inputBuffer.bufferDesriptor.size / 4;
    
    floatView[0] = this.gradThreshold;
    floatView[1] = this.scaleThreshold;
    floatView[2] = this.opacityThreshold;
    uintView[3] = caplen;
    uintView[4] = this.IterationsCount;

    await this.denistyParamsBuffer.updateArray(arrayBuffer);

    await this.cameraBuffer.update(cameraData);
    await this.counterBuffer.updateInteger(new Uint32Array([0]));
    await this.iteratorBuffer.updateInteger(new Uint32Array([this.IterationsCount]));
    await this.freeSlotBuffer.updateInteger(new Uint32Array([this.gaussainsCount]));

    this.commandEncoder.clearBuffer(this.tileBuffer.buffer);
    this.commandEncoder.clearBuffer(this.valuesBuffer.buffer);
    this.commandEncoder.clearBuffer(this.keysBuffer.buffer);
    this.commandEncoder.clearBuffer(this.gGradBuffer.buffer);
    
    let tileSizex = Math.ceil(this.canvas.width / 16);
    let tileSizey = Math.ceil(this.canvas.height / 16);

    this.computePassEncoder = this.commandEncoder.beginComputePass({});
    
    // Project points and get matrices
    this.computePassEncoder.setPipeline(this.computePipeline);
    this.computePassEncoder.setBindGroup(0, this.bindGroup);
    this.computePassEncoder.dispatchWorkgroups(Math.floor((this.gaussainsCount + 63) / this.workGroupSize), 1, 1);
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
      console.log(this.globalKeysCount);
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
          { 
            binding: 2, 
            resource: { buffer: this.counterBuffer.buffer }
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
            resource: this.gradientTextureView 
          },
          { 
            binding: 5, 
            resource: this.transmittanceTextureView 
          },
          { 
            binding: 6, 
            resource: { buffer: this.gGradBuffer.buffer }
          },
          { 
            binding: 7, 
            resource: { buffer: this.denistyBuffer.buffer }
          },
        ]
      });
      this.denistyctrlBindGroup = this.device.createBindGroup({
        layout: this.denistyctrlPipeline.getBindGroupLayout(0),
        entries: [
          { 
            binding: 0,
            resource: { buffer: this.inputBuffer.buffer }
          },
          { 
            binding: 1, 
            resource: { buffer: this.adamMBuffer.buffer } 
          },
          { 
            binding: 2, 
            resource: { buffer: this.adamVBuffer.buffer } 
          },
          { 
            binding: 3, 
            resource: { buffer: this.aliveFlagsBuffer.buffer } 
          },
          { 
            binding: 4, 
            resource: { buffer: this.denistyBuffer.buffer } 
          },
          { 
            binding: 5, 
            resource: { buffer: this.freeSlotBuffer.buffer } 
          },
          { 
            binding: 6, 
            resource: { buffer: this.denistyParamsBuffer.buffer } 
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

    this.computePassEncoder = this.commandEncoder.beginComputePass({});
    const totalWorkgroups = Math.floor((this.gaussainsCount + 63) / this.workGroupSize);
    const dispatchX = Math.min(totalWorkgroups, 256);
    const dispatchY = Math.ceil(totalWorkgroups / dispatchX);

    // Fill keys and values
    this.computePassEncoder.setPipeline(this.keysPipeline);
    this.computePassEncoder.setBindGroup(0, this.keysBindGroup);
    this.computePassEncoder.dispatchWorkgroups(dispatchX, dispatchY, 1);

    // Sort keys
    this.radixSortKernel.dispatch(this.computePassEncoder);

    // Get tile ranges
    this.computePassEncoder.setPipeline(this.tilePipeline);
    this.computePassEncoder.setBindGroup(0, this.tileBindGroup);
    
    const totalTileGroups = Math.floor((this.keysBuffer.bufferDesriptor.size / 4 + 255) / 256);
    const tileDispatchX = Math.min(totalTileGroups, 256);
    const tileDispatchY = Math.ceil(totalTileGroups / tileDispatchX);
    this.computePassEncoder.dispatchWorkgroups(tileDispatchX, tileDispatchY, 1);

    // Rasterization
    this.computePassEncoder.setPipeline(this.rasterPipeline);
    this.computePassEncoder.setBindGroup(0, this.rasterBindGroup);
    this.computePassEncoder.dispatchWorkgroups(tileSizex, tileSizey, 1);
    this.computePassEncoder.end();

    if (flag)
    {

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
      this.computePassEncoder.dispatchWorkgroups(Math.floor((this.gaussainsCount + 63) / this.workGroupSize), 1, 1);
      

      if (this.IterationsCount > this.warmupIters && this.IterationsCount < this.stopDensifyIter && this.IterationsCount % this.densifyInterval === 0)
      {
        const workgroups = Math.ceil(this.gaussainsCount / 256);
        this.computePassEncoder.setPipeline(this.denistyctrlPipeline);
        this.computePassEncoder.setBindGroup(0, this.denistyctrlBindGroup);
        this.computePassEncoder.dispatchWorkgroups(workgroups, 1, 1);

        this.computePassEncoder.setPipeline(this.denistyclearPipeline);
        this.computePassEncoder.setBindGroup(0, this.denistyclearBindGroup);
        this.computePassEncoder.dispatchWorkgroups(workgroups, 1, 1);
        
        this.computePassEncoder.end();
        
        this.commandEncoder.copyBufferToBuffer(
          this.freeSlotBuffer.buffer,
          0,
          this.stagingBuffer2.buffer,
          0,
          4
        );
        this.denFlag = true;
      }
      else 
        this.computePassEncoder.end();
      
      
      this.IterationsCount++;
    }
  } /** End of 'drawLow' function */

  private denFlag = false;

  /**
   * @info End render function
   * @param None. 
   * @returns None.
   */
  public async end(flag: boolean) {
    if (!this.commandEncoder || !this.renderingEnabled) {
      this.commandEncoder = null;
      return;
    }
    if (!this.renderTextureView || !this.gradientTextureView || !this.displayPipeline) {
      this.commandEncoder = null;
      return;
    }
    this.FrameID++;

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
    await this.drawLow(flag);

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
    await this.device.queue.onSubmittedWorkDone();

    if (this.denFlag)
    {
      await this.stagingBuffer2.buffer.mapAsync(GPUMapMode.READ);
      const buf = this.stagingBuffer2.buffer.getMappedRange();
      const view = new Uint32Array(buf);
      
      const newlen = view[0];
      this.stagingBuffer2.buffer.unmap();
      this.denFlag = false;
      this.gaussainsCount = newlen;
      if (newlen > this.inputBuffer.bufferDesriptor.size / 4)
        this.gaussainsCount = this.inputBuffer.bufferDesriptor.size / 4;
      // if (this.gaussainsCount > this.inputBuffer.bufferDesriptor.size / 4)
      // {
      //   const newLen = this.calcCapacity(this.gaussainsCount);
        
      //   const oldInputBuffer = this.inputBuffer.buffer;
      //   const oldOutputBuffer = this.outputBuffer.buffer;
      //   const oldAdamMBuffer = this.adamMBuffer.buffer;
      //   const oldAdamVBuffer = this.adamVBuffer.buffer;
      //   const oldAliveFlagsBuffer = this.aliveFlagsBuffer.buffer;
      //   const oldDenistyBuffer = this.denistyBuffer.buffer;
      //   const oldGGradBuffer = this.gGradBuffer.buffer;

      //   const oldInputSize = oldInputBuffer.size;
      //   const oldOutputSize = oldOutputBuffer.size;
      //   const oldAdamMSize = oldAdamMBuffer.size;
      //   const oldAdamVSize = oldAdamVBuffer.size;
      //   const oldAliveFlagsSize = oldAliveFlagsBuffer.size;
      //   const oldDenistySize = oldDenistyBuffer.size;
      //   const oldGGradSize = oldGGradBuffer.size;

      //   await this.inputBuffer.resize(newLen * 16 * 4);
      //   await this.outputBuffer.resize(newLen * 16 * 4);
      //   await this.gGradBuffer.resize(newLen * 12 * 4);
      //   await this.adamMBuffer.resize(newLen * 16 * 4);
      //   await this.adamVBuffer.resize(newLen * 16 * 4);
      //   await this.aliveFlagsBuffer.resize(newLen * 4);
      //   await this.denistyBuffer.resize(newLen * 2 * 4);

      //   const copyEncoder = this.device.createCommandEncoder();
      //   copyEncoder.copyBufferToBuffer(oldInputBuffer, 0, this.inputBuffer.buffer, 0, oldInputSize);
      //   copyEncoder.copyBufferToBuffer(oldOutputBuffer, 0, this.outputBuffer.buffer, 0, oldOutputSize);
      //   copyEncoder.copyBufferToBuffer(oldAdamMBuffer, 0, this.adamMBuffer.buffer, 0, oldAdamMSize);
      //   copyEncoder.copyBufferToBuffer(oldAdamVBuffer, 0, this.adamVBuffer.buffer, 0, oldAdamVSize);
      //   copyEncoder.copyBufferToBuffer(oldAliveFlagsBuffer, 0, this.aliveFlagsBuffer.buffer, 0, oldAliveFlagsSize);
      //   copyEncoder.copyBufferToBuffer(oldDenistyBuffer, 0, this.denistyBuffer.buffer, 0, oldDenistySize);
      //   copyEncoder.copyBufferToBuffer(oldGGradBuffer, 0, this.gGradBuffer.buffer, 0, oldGGradSize);
      //   this.queue.submit([copyEncoder.finish()]);

      //   oldInputBuffer.destroy();
      //   oldOutputBuffer.destroy();
      //   oldAdamMBuffer.destroy();
      //   oldAdamVBuffer.destroy();
      //   oldAliveFlagsBuffer.destroy();
      //   oldDenistyBuffer.destroy();
      //   oldGGradBuffer.destroy();

      //   this.bindGroup = this.device.createBindGroup({
      //     layout: this.computePipeline.getBindGroupLayout(0),
      //     entries: [
      //       { 
      //         binding: 0,
      //         resource: { buffer: this.cameraBuffer.buffer }
      //       },
      //       { 
      //         binding: 1, 
      //         resource: { buffer: this.inputBuffer.buffer } 
      //       },
      //       { 
      //         binding: 2, 
      //         resource: { buffer: this.outputBuffer.buffer } 
      //       },
      //       { 
      //         binding: 3, 
      //         resource: { buffer: this.counterBuffer.buffer } 
      //       }
      //     ]
      //   });

          
      //   this.keysBindGroup = this.device.createBindGroup({
      //     layout: this.keysPipeline.getBindGroupLayout(0),
      //     entries: [
      //       { 
      //         binding: 0,
      //         resource: { buffer: this.cameraBuffer.buffer }
      //       },
      //       { 
      //         binding: 1, 
      //         resource: { buffer: this.outputBuffer.buffer } 
      //       },
      //       { 
      //         binding: 2, 
      //         resource: { buffer: this.keysBuffer.buffer } 
      //       },
      //       { 
      //         binding: 3, 
      //         resource: { buffer: this.valuesBuffer.buffer } 
      //       },
      //       { 
      //         binding: 4, 
      //         resource: { buffer: this.counterBuffer.buffer } 
      //       }
      //     ]
      //   });

      //   this.rasterBindGroup = this.device.createBindGroup({
      //     layout: this.rasterPipeline.getBindGroupLayout(0),
      //     entries: [
      //       { 
      //         binding: 0,
      //         resource: { buffer: this.cameraBuffer.buffer }
      //       },
      //       { 
      //         binding: 1, 
      //         resource: { buffer: this.outputBuffer.buffer } 
      //       },
      //       { 
      //         binding: 2, 
      //         resource: { buffer: this.valuesBuffer.buffer } 
      //       },
      //       { 
      //         binding: 3, 
      //         resource: { buffer: this.tileBuffer.buffer } 
      //       },
      //       { 
      //         binding: 4, 
      //         resource: this.renderTextureView 
      //       },
      //       { 
      //         binding: 5, 
      //         resource: this.transmittanceTextureView 
      //       },
      //     ]
      //   });
          
      //   this.backwardBindGroup = this.device.createBindGroup({
      //     layout: this.backwardPipeline.getBindGroupLayout(0),
      //     entries: [
      //       { 
      //         binding: 0,
      //         resource: { buffer: this.cameraBuffer.buffer }
      //       },
      //       { 
      //         binding: 1, 
      //         resource: { buffer: this.outputBuffer.buffer } 
      //       },
      //       { 
      //         binding: 2, 
      //         resource: { buffer: this.valuesBuffer.buffer } 
      //       },
      //       { 
      //         binding: 3, 
      //         resource: { buffer: this.tileBuffer.buffer } 
      //       },
      //       { 
      //         binding: 4, 
      //         resource: this.gradientTextureView 
      //       },
      //       { 
      //         binding: 5, 
      //         resource: this.transmittanceTextureView 
      //       },
      //       { 
      //         binding: 6, 
      //         resource: { buffer: this.gGradBuffer.buffer }
      //       },
      //       { 
      //         binding: 7, 
      //         resource: { buffer: this.denistyBuffer.buffer }
      //       },
      //     ]
      //   });

      //   this.learnBindGroup = this.device.createBindGroup({
      //     layout: this.learnPipeline.getBindGroupLayout(0),
      //     entries: [
      //       { 
      //         binding: 0,
      //         resource: { buffer: this.cameraBuffer.buffer }
      //       },
      //       { 
      //         binding: 1, 
      //         resource: { buffer: this.inputBuffer.buffer } 
      //       },
      //       { 
      //         binding: 2, 
      //         resource: { buffer: this.gGradBuffer.buffer } 
      //       },
      //       { 
      //         binding: 3, 
      //         resource: { buffer: this.adamMBuffer.buffer } 
      //       },
      //       { 
      //         binding: 4, 
      //         resource: { buffer: this.adamVBuffer.buffer } 
      //       },
      //       { 
      //         binding: 5, 
      //         resource: { buffer: this.iteratorBuffer.buffer } 
      //       },
      //     ]
      //   });

      //   this.denistyclearBindGroup = this.device.createBindGroup({
      //     layout: this.denistyclearPipeline.getBindGroupLayout(0),
      //     entries: [
      //       { 
      //         binding: 0,
      //         resource: { buffer: this.denistyBuffer.buffer }
      //       },
      //     ]
      //   });
      //   this.denistyctrlBindGroup = this.device.createBindGroup({
      //     layout: this.denistyctrlPipeline.getBindGroupLayout(0),
      //     entries: [
      //       { 
      //         binding: 0,
      //         resource: { buffer: this.inputBuffer.buffer }
      //       },
      //       { 
      //         binding: 1, 
      //         resource: { buffer: this.adamMBuffer.buffer } 
      //       },
      //       { 
      //         binding: 2, 
      //         resource: { buffer: this.adamVBuffer.buffer } 
      //       },
      //       { 
      //         binding: 3, 
      //         resource: { buffer: this.aliveFlagsBuffer.buffer } 
      //       },
      //       { 
      //         binding: 4, 
      //         resource: { buffer: this.denistyBuffer.buffer } 
      //       },
      //       { 
      //         binding: 5, 
      //         resource: { buffer: this.freeSlotBuffer.buffer } 
      //       },
      //       { 
      //         binding: 6, 
      //         resource: { buffer: this.denistyParamsBuffer.buffer } 
      //       },
      //     ]
      //   });
      // }
      
      const len = this.calcCapacity(this.gaussainsCount);

      const aliveArrayBuffer = new ArrayBuffer(len * 4);
      const aliveView = new Uint32Array(aliveArrayBuffer);

      for (let i = 0; i < this.gaussainsCount; i++) {
        aliveView[i] = 1;
      }

      await this.aliveFlagsBuffer.resize(len * 4);
      await this.aliveFlagsBuffer.updateArray(aliveArrayBuffer);

      const densityArrayBuffer = new ArrayBuffer(len * 2 * 4);

      await this.denistyBuffer.resize(len * 2 * 4);
      await this.denistyBuffer.updateArray(densityArrayBuffer);
    }
    
    this.passEncoder = null;
    this.commandEncoder = null;
    input.clear();
    // this.primitives = [];
  } /** End of 'end' function */

  public async cleanup() {
    this.renderingEnabled = false;

    // Wait for any pending GPU operations
    await this.device.queue.onSubmittedWorkDone();
    this.commandEncoder = null;
    this.passEncoder = null;
    this.computePassEncoder = null;

    // Destroy all textures
    if (this.renderTexture) {
      this.renderTexture.destroy();
      this.renderTexture = null;
    }
    if (this.renderTextureView) {
      this.renderTextureView = null;
    }
    if (this.transmittanceTexture) {
      this.transmittanceTexture.destroy();
      this.transmittanceTexture = null;
    }
    if (this.transmittanceTextureView) {
      this.transmittanceTextureView = null;
    }
    if (this.gradientTexture) {
      this.gradientTexture.destroy();
      this.gradientTexture = null;
    }
    if (this.gradientTextureView) {
      this.gradientTextureView = null;
    }
    if (this.imageTexture) {
      this.imageTexture.destroy();
      this.imageTexture = null;
    }
    if (this.imageTextureView) {
      this.imageTextureView = null;
    }

    // Destroy all buffers
    const buffers = [
      this.inputBuffer,
      this.outputBuffer,
      this.keysBuffer,
      this.valuesBuffer,
      this.tileBuffer,
      this.counterBuffer,
      this.stagingBuffer,
      this.stagingBuffer2,
      this.cameraBuffer,
      this.gGradBuffer,
      this.adamMBuffer,
      this.adamVBuffer,
      this.iteratorBuffer,
      this.freeSlotBuffer,
      this.aliveFlagsBuffer,
      this.denistyBuffer,
      this.denistyParamsBuffer
    ];

    for (const buf of buffers) {
      if (buf && buf.buffer) {
        buf.buffer.destroy();
      }
    }

    // Clear primitives
    this.primitives = [];
    this.gaussainsCount = 0;
    this.globalKeysCount = 1;
    this.IterationsCount = 1;
    this.FrameID = 0;
    this.denFlag = false;
    this.cameraData = undefined;

    // Nullify all bind groups
    this.bindGroup = null as any;
    this.tileBindGroup = null as any;
    this.keysBindGroup = null as any;
    this.rasterBindGroup = null as any;
    this.displayBindGroup = null as any;
    this.gradientBindGroup = null as any;
    this.backwardBindGroup = null as any;
    this.learnBindGroup = null as any;
    this.denistyclearBindGroup = null as any;
    this.denistyctrlBindGroup = null as any;

    // Nullify pipelines
    this.computePipeline = null as any;
    this.keysPipeline = null as any;
    this.tilePipeline = null as any;
    this.rasterPipeline = null as any;
    this.displayPipeline = null as any;
    this.gradientPipeline = null as any;
    this.backwardPipeline = null as any;
    this.learnPipeline = null as any;
    this.denistyclearPipeline = null as any;
    this.denistyctrlPipeline = null as any;
  }

  public async reinit(id: Element) {
    await this.cleanup();
    await this.init(id);
  }
} /** End of 'render' class */

export { render }; 

/** END OF 'render.ts' FILE */