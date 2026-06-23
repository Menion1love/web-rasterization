export interface SceneDefaults {
  /** Default scale for JSON point clouds (pos + color only). */
  jsonScale: number;
  jsonOpacity: number;
  jsonRotation: [number, number, number, number];
  /** Clamp scale components when loading .bin (PLY export uses exp scale). */
  binScaleMax: number;
}

export interface PathsConfig {
  /** Directory with training target images, relative to site root. */
  imagesDir: string;
}

export interface TrainingUIConfig {
  /** Switch random camera every N frames when auto-train is on. */
  autoCameraSwitchInterval: number;
}

export interface RenderTrainingConfig {
  warmupIters: number;
  densifyInterval: number;
  stopDensifyIter: number;
  gradThreshold: number;
  scaleThreshold: number;
  opacityThreshold: number;
}

export interface AppConfig {
  scene: SceneDefaults;
  paths: PathsConfig;
  trainingUI: TrainingUIConfig;
  renderTraining: RenderTrainingConfig;
}

export const DEFAULT_CONFIG: AppConfig = {
  scene: {
    jsonScale: 0.02,
    jsonOpacity: 0.6,
    jsonRotation: [0, 0, 0, 1],
    binScaleMax: 1.0,
  },
  paths: {
    imagesDir: 'bin/images',
  },
  trainingUI: {
    autoCameraSwitchInterval: 100,
  },
  renderTraining: {
    warmupIters: 200,
    densifyInterval: 100,
    stopDensifyIter: 15000,
    gradThreshold: 0.003,
    scaleThreshold: 0.01,
    opacityThreshold: 0.05,
  },
};
