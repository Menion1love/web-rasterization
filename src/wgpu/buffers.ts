/***************************************************************
 * Copyright (C) 2026
 *    Computer Graphics Support Group of 30 Phys-Math Lyceum
 ***************************************************************/

/** FILE NAME   : buffers.ts
 *  PURPOSE     : Gaussian splatting project.
 *  PROGRAMMER  : CGSG'An'2026.
 *                Timofey Hudyakov (TH4).
 *  LAST UPDATE : 08.06.2026
 * 
 *  No part of this file may be changed without agreement of
 *  Computer Graphics Support Group of 30 Phys-Math Lyceum
 */


import { core } from './core.ts'

/** Buffer interface */
interface buffer_descriptor {
  usage: GPUBufferUsageFlags;
  size: number;
  label?: string;
  type?: GPUBufferBindingType;
  data?: Float32Array;
} /** End of 'buffer_descriptor' interface */

/** Buffer class */
class buffer {
  /** #private parameters */
  public bufferDesriptor!: GPUBufferDescriptor;
  private render: core;
  
  /**
   * @info Shader constructor
   * @param render: render instance
   */
  constructor(renderInstance: core) {
    this.render = renderInstance;
  } /** End of 'constructor' function */

  /** #public parameters */
  public buffer!: GPUBuffer;
  public bufferType!: GPUBufferBindingType;
  public isSizeChanged: boolean = false;

  /**
   * @info Create buffer function
   * @param bufferParams: buffer_descriptor
   * @returns none
   */
  public async create(bufferParams: buffer_descriptor) {
    if (bufferParams.type) this.bufferType = bufferParams.type;
    if (bufferParams.data) {
      this.bufferDesriptor = {
        size: bufferParams.size,
        usage: bufferParams.usage,
        mappedAtCreation: true,
        label: bufferParams.label
      };
      this.buffer = this.render.device.createBuffer(this.bufferDesriptor);
      new Float32Array(this.buffer.getMappedRange()).set(bufferParams.data);
      this.buffer.unmap();
    }
    else {
      this.bufferDesriptor = {
        size: bufferParams.size,
        usage: bufferParams.usage,
        label: bufferParams.label
      };  
      this.buffer = this.render.device.createBuffer(this.bufferDesriptor);
    }
  } /** End of 'create' function */

  /**
   * @info Resize buffer function with growth strategy
   * @param newSize: number
   * @returns None.
   */
  public async resize(newSize: number, doublesize: boolean = false ) {
    if (this.bufferDesriptor.size < newSize)
    {
      await this.destroy();
  
      if (doublesize)
        this.bufferDesriptor.size = newSize * 2;
      else
        this.bufferDesriptor.size = newSize;

      this.buffer = this.render.device.createBuffer(this.bufferDesriptor);
      this.isSizeChanged = true;
    }
  } /** End of 'resize' function */

  /**
   * @info Update buffer function
   * @param data: Float32Array  
   * @returns None.
   */
  public async update(data: Float32Array) {
    await this.resize(data.byteLength);
    this.render.queue.writeBuffer(this.buffer, 0, data as GPUAllowSharedBufferSource);
  } /** End of 'update' function */

  /**
   * @info Update buffer function
   * @param data: Float32Array  
   * @returns None.
   */
  public async updateInteger(data: Uint32Array) {
    await this.resize(data.byteLength);
    this.render.queue.writeBuffer(this.buffer, 0, data as GPUAllowSharedBufferSource);
  } /** End of 'update' function */

  /**
   * @info Update buffer function
   * @param data: Float32Array  
   * @returns None.
   */
  public async updateArray(data: ArrayBuffer) {
    await this.resize(data.byteLength);
    this.render.queue.writeBuffer(this.buffer, 0, data as GPUAllowSharedBufferSource);
  } /** End of 'update' function */

  /**
   * @info Copy buffer function
   * @param encoder: GPUCommandEncoder
   * @param buffer: buffer
   * @returns None.
   */
  public async copy(encoder: GPUCommandEncoder, buffer: buffer) {
    await this.resize(buffer.bufferDesriptor.size);
    encoder.copyBufferToBuffer(buffer.buffer, 0, this.buffer, 0, buffer.bufferDesriptor.size);
  } /** End of 'copy' function */

  /**
   * @info Destroy buffer function
   * @returns None.
   */
  public async destroy() {
    if (this.buffer != null)
    {
      this.buffer.destroy();
      this.buffer = null as unknown as GPUBuffer;
    }
  } /** End of 'destroy' function */
} /** End of 'buffer' class */

/** EXPORTS */
export { buffer };

/** END OF 'buffers.ts' FILE */
