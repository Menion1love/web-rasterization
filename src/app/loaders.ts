import { primitive } from '../render.ts';
import { vec3 } from '../mth/mth_vec3.ts';
import { vec4 } from '../mth/mth_vec4.ts';
import type { AppConfig } from './config.ts';
import type { CameraData, GaussianData } from './types.ts';

export async function loadPointsFile(
  file: File,
  config: AppConfig,
): Promise<primitive[]> {
  const extension = file.name.split('.').pop()?.toLowerCase();

  if (extension === 'json') {
    return loadPointsFromJSON(file, config);
  }
  if (extension === 'bin') {
    return loadPointsFromBinary(file, config);
  }

  throw new Error(`Unsupported points format: ${extension ?? 'unknown'}`);
}

async function loadPointsFromJSON(file: File, config: AppConfig): Promise<primitive[]> {
  const text = await file.text();
  const gaussiansData: GaussianData[] = JSON.parse(text);
  const { jsonScale, jsonOpacity, jsonRotation } = config.scene;

  const primitives: primitive[] = [];
  for (let i = 0; i < gaussiansData.length; i++) {
    const g = gaussiansData[i];
    const prim = new primitive(
      new vec3(g.pos[0], g.pos[1], g.pos[2]),
      new vec3(jsonScale),
      new vec4(jsonRotation[0], jsonRotation[1], jsonRotation[2], jsonRotation[3]),
      jsonOpacity,
      new vec4(g.color[0], g.color[1], g.color[2], g.color[3]),
    );
    prim.index = i;
    primitives.push(prim);
  }

  return primitives;
}

async function loadPointsFromBinary(file: File, config: AppConfig): Promise<primitive[]> {
  const arrayBuffer = await file.arrayBuffer();
  const floatData = new Float32Array(arrayBuffer);

  const stride = 14;
  if (floatData.length % stride !== 0) {
    throw new Error(`Invalid .bin size: ${floatData.length} floats (expected multiple of ${stride})`);
  }

  const numPrimitives = floatData.length / stride;
  const maxScale = config.scene.binScaleMax;
  const primitives: primitive[] = [];

  for (let i = 0; i < numPrimitives; i++) {
    const offset = i * stride;

    const s0 = floatData[offset + 3];
    const s1 = floatData[offset + 4];
    const s2 = floatData[offset + 5];

    primitives.push(new primitive(
      new vec3(floatData[offset + 0], floatData[offset + 1], floatData[offset + 2]),
      new vec3(
        s0 > maxScale ? maxScale : s0,
        s1 > maxScale ? maxScale : s1,
        s2 > maxScale ? maxScale : s2,
      ),
      new vec4(
        floatData[offset + 6],
        floatData[offset + 7],
        floatData[offset + 8],
        floatData[offset + 9],
      ),
      floatData[offset + 10],
      new vec4(
        floatData[offset + 11],
        floatData[offset + 12],
        floatData[offset + 13],
        1.0,
      ),
    ));
  }

  return primitives;
}

export async function loadCamerasFile(file: File): Promise<CameraData[]> {
  const text = await file.text();
  const cameras: CameraData[] = JSON.parse(text);

  if (!Array.isArray(cameras) || cameras.length === 0) {
    throw new Error('Cameras file must be a non-empty JSON array');
  }

  return cameras;
}

export function imagePathForCamera(config: AppConfig, camera: CameraData): string {
  return `${config.paths.imagesDir}/${camera.name}`;
}
