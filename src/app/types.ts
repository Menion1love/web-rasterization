export type AppPhase = 'idle' | 'view' | 'train';

export type PointsFormat = 'json' | 'bin';

export interface CameraData {
  name: string;
  loc: number[];
  at: number[];
  up: number[];
}

export interface GaussianData {
  pos: number[];
  color: number[];
}
