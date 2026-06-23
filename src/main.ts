import './style.css';
import { render } from './render.ts';
import { vec3 } from './mth/mth_vec3.ts';
import { DEFAULT_CONFIG, type AppConfig } from './app/config.ts';
import { imagePathForCamera, loadCamerasFile, loadPointsFile } from './app/loaders.ts';
import type { AppPhase, CameraData, PointsFormat } from './app/types.ts';
import type { primitive } from './render.ts';

class GaussianApp {
  private config: AppConfig;
  private rnd: render;
  private phase: AppPhase = 'idle';

  private primitives: primitive[] = [];
  private cameras: CameraData[] = [];
  private currentCam = 0;
  private pointsFileName = '';
  private camerasFileName = '';
  private pointsFormat: PointsFormat | null = null;

  private pointsLoaded = false;
  private camerasLoaded = false;
  private sceneReady = false;
  private isTrainingActive = false;
  private trainingMode: 'manual' | 'auto' = 'manual';
  private isResetting = false;

  private statusEl!: HTMLElement;
  private loadPanelEl!: HTMLElement;
  private trainingControlsEl!: HTMLElement;
  private phaseBadgeEl!: HTMLElement;
  private pointsFileInput!: HTMLInputElement;
  private camerasFileInput!: HTMLInputElement;
  private loadPointsBtn!: HTMLButtonElement;
  private loadCamerasBtn!: HTMLButtonElement;
  private pointsStateEl!: HTMLElement;
  private camerasStateEl!: HTMLElement;
  private pointsDotEl!: HTMLElement;
  private camerasDotEl!: HTMLElement;
  private prevBtn!: HTMLButtonElement;
  private nextBtn!: HTMLButtonElement;
  private counterDisplay!: HTMLSpanElement;
  private trainingToggleBtn!: HTMLButtonElement;
  private autoTrainBtn!: HTMLButtonElement;

  constructor(config: AppConfig = DEFAULT_CONFIG) {
    this.config = config;
    this.rnd = new render();
  }

  async init() {
    const canvas = document.querySelector('#webgpu-canvas') as HTMLElement;
    await this.rnd.init(canvas);
    this.rnd.applyTrainingConfig(this.config.renderTraining);
    this.bindUI();
    this.clearSceneData();
    this.setPhase('idle');
    this.startRenderLoop();
  }

  private bindUI() {
    this.statusEl = document.getElementById('statusBar') as HTMLElement;
    this.loadPanelEl = document.getElementById('loadPanel') as HTMLElement;
    this.trainingControlsEl = document.getElementById('trainingControls') as HTMLElement;
    this.phaseBadgeEl = document.getElementById('phaseBadge') as HTMLElement;
    this.pointsFileInput = document.getElementById('pointsFileInput') as HTMLInputElement;
    this.camerasFileInput = document.getElementById('camerasFileInput') as HTMLInputElement;
    this.loadPointsBtn = document.getElementById('loadPointsBtn') as HTMLButtonElement;
    this.loadCamerasBtn = document.getElementById('loadCamerasBtn') as HTMLButtonElement;
    this.pointsStateEl = document.getElementById('pointsState') as HTMLElement;
    this.camerasStateEl = document.getElementById('camerasState') as HTMLElement;
    this.pointsDotEl = document.getElementById('pointsDot') as HTMLElement;
    this.camerasDotEl = document.getElementById('camerasDot') as HTMLElement;
    this.prevBtn = document.getElementById('prevBtn') as HTMLButtonElement;
    this.nextBtn = document.getElementById('nextBtn') as HTMLButtonElement;
    this.counterDisplay = document.getElementById('counterValue') as HTMLSpanElement;
    this.trainingToggleBtn = document.getElementById('trainingToggleBtn') as HTMLButtonElement;
    this.autoTrainBtn = document.getElementById('autoTrainBtn') as HTMLButtonElement;

    document.getElementById('loadPointsBtn')?.addEventListener('click', () => {
      if (this.pointsLoaded) return;
      this.pointsFileInput.click();
    });
    this.loadCamerasBtn.addEventListener('click', () => {
      if (!this.canTrain() || this.camerasLoaded) return;
      this.camerasFileInput.click();
    });
    document.getElementById('resetSceneBtn')?.addEventListener('click', () => void this.resetScene());

    this.pointsFileInput.addEventListener('change', (e) => void this.onPointsSelected(e));
    this.camerasFileInput.addEventListener('change', (e) => void this.onCamerasSelected(e));

    this.prevBtn.addEventListener('click', () => this.prevCamera());
    this.nextBtn.addEventListener('click', () => this.nextCamera());
    this.trainingToggleBtn.addEventListener('click', () => this.toggleTraining());
    this.autoTrainBtn.addEventListener('click', () => this.toggleAutoTraining());
  }

  private setStatus(text: string) {
    if (this.statusEl) {
      this.statusEl.textContent = text;
    }
  }

  private clearSceneData() {
    this.primitives = [];
    this.cameras = [];
    this.currentCam = 0;
    this.pointsFileName = '';
    this.camerasFileName = '';
    this.pointsFormat = null;
    this.pointsLoaded = false;
    this.camerasLoaded = false;
    this.isTrainingActive = false;
    this.trainingMode = 'manual';
  }

  private canTrain(): boolean {
    return this.pointsLoaded && this.pointsFormat === 'json';
  }

  private detectPointsFormat(file: File): PointsFormat | null {
    const extension = file.name.split('.').pop()?.toLowerCase();
    if (extension === 'json') return 'json';
    if (extension === 'bin') return 'bin';
    return null;
  }

  private clearCamerasData() {
    this.cameras = [];
    this.camerasFileName = '';
    this.camerasLoaded = false;
    this.currentCam = 0;
    this.isTrainingActive = false;
    this.trainingMode = 'manual';
    this.camerasFileInput.value = '';
  }

  private updateAssetState() {
    const trainingAllowed = this.canTrain();

    this.loadPointsBtn.disabled = this.pointsLoaded;
    this.loadPointsBtn.hidden = this.pointsLoaded;

    this.loadCamerasBtn.disabled = !trainingAllowed || this.camerasLoaded;
    this.loadCamerasBtn.hidden = (!trainingAllowed && this.pointsLoaded) || this.camerasLoaded;

    if (this.pointsLoaded) {
      const formatLabel = this.pointsFormat === 'bin' ? ' · viewer only' : '';
      this.pointsStateEl.textContent = `${this.pointsFileName || 'Points loaded'}${formatLabel}`;
      this.pointsDotEl.classList.add('is-loaded');
    } else {
      this.pointsStateEl.textContent = 'Points not loaded';
      this.pointsDotEl.classList.remove('is-loaded');
      this.loadPointsBtn.hidden = false;
      this.loadCamerasBtn.hidden = false;
    }

    if (!trainingAllowed && this.pointsLoaded) {
      this.camerasStateEl.textContent = 'Training unavailable for .bin';
      this.camerasDotEl.classList.remove('is-loaded');
    } else if (this.camerasLoaded && this.cameras.length > 0) {
      this.camerasStateEl.textContent = `${this.camerasFileName || 'Cameras'} · ${this.cameras.length} views`;
      this.camerasDotEl.classList.add('is-loaded');
    } else if (this.pointsLoaded && trainingAllowed) {
      this.camerasStateEl.textContent = 'Cameras not loaded';
      this.camerasDotEl.classList.remove('is-loaded');
    } else {
      this.camerasStateEl.textContent = 'Cameras not loaded';
      this.camerasDotEl.classList.remove('is-loaded');
    }
  }

  private updatePhaseBadge() {
    this.phaseBadgeEl.textContent = this.phase;
    this.phaseBadgeEl.dataset.phase = this.phase;
  }

  private setPhase(phase: AppPhase) {
    this.phase = phase;
    this.sceneReady = phase !== 'idle';

    const showTraining = phase === 'train';
    this.loadPanelEl.classList.toggle('is-scene-active', this.sceneReady);
    this.trainingControlsEl.classList.toggle('is-visible', showTraining);

    this.updatePhaseBadge();
    this.updateAssetState();
    this.updateTrainingButtons();
    this.updateCameraDisplay();

    if (phase === 'idle') {
      this.setStatus('Load points: .json for training, .bin for viewer only.');
    } else if (phase === 'view') {
      if (this.pointsFormat === 'bin') {
        this.setStatus('Viewer mode (.bin) — reset scene to load another file.');
      } else if (this.camerasLoaded) {
        this.setStatus('Training ready. Prev/Next or Auto to switch camera views.');
      } else {
        this.setStatus('Load cameras to train, or reset scene to load new points.');
      }
    } else {
      this.setStatus('Training active. Reset scene to load a different dataset.');
    }
  }

  private updateTrainingButtons() {
    this.trainingToggleBtn.textContent = this.isTrainingActive ? 'Stop' : 'Train';
    this.trainingToggleBtn.dataset.active = this.isTrainingActive ? 'true' : 'false';

    this.autoTrainBtn.textContent = this.trainingMode === 'auto' ? 'Auto: ON' : 'Auto: OFF';
    this.autoTrainBtn.dataset.active = this.trainingMode === 'auto' ? 'true' : 'false';

    const trainingEnabled = this.phase === 'train' && this.camerasLoaded;
    this.prevBtn.disabled = !trainingEnabled;
    this.nextBtn.disabled = !trainingEnabled;
    this.trainingToggleBtn.disabled = !trainingEnabled;
    this.autoTrainBtn.disabled = !trainingEnabled;
  }

  private clearFileInputs() {
    this.pointsFileInput.value = '';
    this.camerasFileInput.value = '';
  }

  private async onPointsSelected(event: Event) {
    if (this.isResetting) return;

    if (this.pointsLoaded) {
      this.setStatus('Points already loaded. Reset scene to load a different file.');
      (event.target as HTMLInputElement).value = '';
      return;
    }

    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    this.setStatus(`Loading points: ${file.name}...`);

    try {
      this.clearCamerasData();

      const format = this.detectPointsFormat(file);
      if (!format) {
        throw new Error('Unsupported file format');
      }

      this.primitives = await loadPointsFile(file, this.config);
      this.pointsLoaded = true;
      this.pointsFileName = file.name;
      this.pointsFormat = format;
      await this.rnd.loadScene(this.primitives);

      this.setPhase(this.camerasLoaded && this.canTrain() ? 'train' : 'view');
    } catch (error) {
      console.error('Failed to load points:', error);
      this.setStatus('Failed to load points. Check format and try again.');
      this.pointsLoaded = false;
      this.pointsFileName = '';
      this.pointsFormat = null;
      this.primitives = [];
      this.updateAssetState();
    }
  }

  private async onCamerasSelected(event: Event) {
    if (this.isResetting) return;

    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    if (!this.pointsLoaded) {
      this.setStatus('Load points first, then cameras.');
      input.value = '';
      return;
    }

    if (this.camerasLoaded) {
      this.setStatus('Cameras already loaded. Reset scene to load a different set.');
      input.value = '';
      return;
    }

    if (!this.canTrain()) {
      this.setStatus('Training is only available for .json point clouds.');
      input.value = '';
      return;
    }

    this.setStatus(`Loading cameras: ${file.name}...`);

    try {
      this.cameras = await loadCamerasFile(file);
      this.camerasLoaded = true;
      this.camerasFileName = file.name;
      this.currentCam = 0;

      await this.rnd.loadScene(this.primitives);
      await this.loadCurrentImage();

      this.setPhase('train');
    } catch (error) {
      console.error('Failed to load cameras:', error);
      this.setStatus('Failed to load cameras. Expected JSON array with loc/at/up/name.');
      this.clearCamerasData();
      this.updateAssetState();
      this.updateCameraDisplay();
    }
  }

  private async loadCurrentImage() {
    if (!this.camerasLoaded || this.cameras.length === 0) return;

    const targetW = this.rnd.canvas.width;
    const targetH = this.rnd.canvas.height;
    const cam = this.cameras[this.currentCam];
    const path = imagePathForCamera(this.config, cam);

    try {
      const response = await fetch(path);
      if (!response.ok) {
        console.error(`Failed to load image: ${path}`);
        this.setStatus(`Camera image not found: ${path}`);
        return;
      }

      const blob = await response.blob();
      if (blob.size === 0) return;

      const imageBitmap = await createImageBitmap(blob, {
        colorSpaceConversion: 'default',
        resizeWidth: targetW,
        resizeHeight: targetH,
        resizeQuality: 'high',
      });

      const imageTexture = this.rnd.imageTexture;
      if (!imageTexture) return;

      this.rnd.device.queue.copyExternalImageToTexture(
        { source: imageBitmap },
        { texture: imageTexture },
        [imageBitmap.width, imageBitmap.height],
      );

      this.rnd.imageTextureView = imageTexture.createView();
      this.rnd.reload();
    } catch (error) {
      console.error(`Error loading image ${path}:`, error);
      this.setStatus(`Error loading image: ${path}`);
    }
  }

  private prevCamera() {
    if (!this.camerasLoaded || this.cameras.length === 0) return;
    this.currentCam = (this.currentCam - 1 + this.cameras.length) % this.cameras.length;
    this.updateCameraDisplay();
    void this.loadCurrentImage();
  }

  private nextCamera() {
    if (!this.camerasLoaded || this.cameras.length === 0) return;
    this.currentCam = (this.currentCam + 1) % this.cameras.length;
    this.updateCameraDisplay();
    void this.loadCurrentImage();
  }

  private toggleTraining() {
    if (this.phase !== 'train' || !this.camerasLoaded) return;
    this.isTrainingActive = !this.isTrainingActive;
    this.updateTrainingButtons();
  }

  private toggleAutoTraining() {
    if (this.phase !== 'train' || !this.camerasLoaded) return;
    this.trainingMode = this.trainingMode === 'auto' ? 'manual' : 'auto';
    this.updateTrainingButtons();
  }

  private updateCameraDisplay() {
    if (!this.camerasLoaded || this.cameras.length === 0) {
      this.counterDisplay.textContent = '—';
      return;
    }
    this.counterDisplay.textContent = `${this.currentCam + 1} / ${this.cameras.length}`;
  }

  async resetScene() {
    if (this.isResetting) return;
    this.isResetting = true;

    this.clearSceneData();
    this.clearFileInputs();
    this.setPhase('idle');

    try {
      const canvas = document.querySelector('#webgpu-canvas') as HTMLElement;
      await this.rnd.reinit(canvas);
      this.rnd.applyTrainingConfig(this.config.renderTraining);
      this.setStatus('Scene cleared. Load a new points file to start.');
    } finally {
      this.isResetting = false;
    }
  }

  private startRenderLoop() {
    let frame = 0;

    const draw = async () => {
      if (this.sceneReady && !this.isResetting) {
        await this.rnd.start();

        if (this.phase === 'train' && this.isTrainingActive && this.cameras.length > 0) {
          const interval = this.config.trainingUI.autoCameraSwitchInterval;
          if (this.trainingMode === 'auto' && frame % interval === 0) {
            this.currentCam = Math.floor(Math.random() * this.cameras.length);
            this.updateCameraDisplay();
            void this.loadCurrentImage();
          }

          const cam = this.cameras[this.currentCam];
          const loc = new vec3(cam.loc[0], cam.loc[1], cam.loc[2]);
          const at = new vec3(cam.at[0], cam.at[1], cam.at[2]);
          const up = new vec3(cam.up[0], cam.up[1], cam.up[2]);
          this.rnd.controls.cam.set(loc, at.add(loc), up);
        }

        await this.rnd.end(this.isTrainingActive && this.phase === 'train');
      }

      frame++;
      window.requestAnimationFrame(draw);
    };

    draw();
  }
}

void (async () => {
  const app = new GaussianApp();
  await app.init();
})();
