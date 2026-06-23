import './style.css'
import { render, primitive } from "./render.ts"
import { vec3 } from "./mth/mth_vec3.ts"
import { vec4 } from "./mth/mth_vec4.ts"

type AppMode = 'none' | 'viewer' | 'training';
type MenuState = 'main' | 'points' | 'cameras';

interface CameraData {
  name: string;
  loc: number[];
  at: number[];
  up: number[];
}

interface GaussianData {
  pos: number[];
  color: number[];
}

class GaussianViewer {
  rnd: render;
  mode: AppMode;
  primitives: primitive[] = [];
  
  cameras: CameraData[] = [];
  currentCam: number = 0;
  trainingMode: 'manual' | 'auto' = 'manual';
  
  prevBtn!: HTMLButtonElement;
  nextBtn!: HTMLButtonElement;
  counterDisplay!: HTMLSpanElement;
  trainingToggleBtn!: HTMLButtonElement;
  autoTrainBtn!: HTMLButtonElement;
  
  pointsFileInput!: HTMLInputElement;
  camerasFileInput!: HTMLInputElement;
  
  pointsLoaded: boolean = false;
  camerasLoaded: boolean = false;
  isTrainingActive: boolean = false;
  isReady: boolean = false;
  
  menuState: MenuState = 'main';
  
  constructor() {
    this.rnd = new render();
    this.mode = 'none';
  }

  async init() {
    await this.rnd.init(document.querySelector("#webgpu-canvas") as HTMLElement);
    this.setupUI();
    this.drawMessage('LOAD POINTS TO START');
    this.startRenderLoop();
  }

  drawMessage(text: string) {
    const canvas = this.rnd.canvas as HTMLCanvasElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#ffffff';
    ctx.font = '24px monospace';
    ctx.textAlign = 'center';
    
    const lines = text.split('\n');
    lines.forEach((line, i) => {
      ctx.fillText(line, canvas.width / 2, canvas.height / 2 + i * 30);
    });
  }

  setupUI() {
    this.prevBtn = document.getElementById('prevBtn') as HTMLButtonElement;
    this.nextBtn = document.getElementById('nextBtn') as HTMLButtonElement;
    this.counterDisplay = document.getElementById('counterValue') as HTMLSpanElement;
    this.trainingToggleBtn = document.getElementById('trainingToggleBtn') as HTMLButtonElement;
    this.autoTrainBtn = document.getElementById('autoTrainBtn') as HTMLButtonElement;
    
    this.pointsFileInput = document.getElementById('pointsFileInput') as HTMLInputElement;
    this.camerasFileInput = document.getElementById('camerasFileInput') as HTMLInputElement;
    
    this.prevBtn?.addEventListener('click', () => this.prevCamera());
    this.nextBtn?.addEventListener('click', () => this.nextCamera());
    this.trainingToggleBtn?.addEventListener('click', () => this.toggleTraining());
    this.autoTrainBtn?.addEventListener('click', () => this.toggleAutoTraining());
    
    this.pointsFileInput?.addEventListener('change', (e) => this.loadPointsFile(e));
    this.camerasFileInput?.addEventListener('change', (e) => this.loadCamerasFile(e));
    
    document.getElementById('menuLoadPoints')?.addEventListener('click', () => this.showPointsMenu());
    document.getElementById('menuLoadCameras')?.addEventListener('click', () => this.showCamerasMenu());
    document.getElementById('menuBackPoints')?.addEventListener('click', () => this.showMainMenu());
    document.getElementById('menuBackCameras')?.addEventListener('click', () => this.showMainMenu());
    document.getElementById('loadPointsBtn')?.addEventListener('click', () => this.pointsFileInput?.click());
    document.getElementById('loadCamerasBtn')?.addEventListener('click', () => this.camerasFileInput?.click());
    document.getElementById('menuClean')?.addEventListener('click', () => this.cleanAndReset());
    
    this.showMainMenu();
    this.updateUIForMode();
  }

  async cleanAndReset() {
    this.isReady = false;
    this.isTrainingActive = false;
    this.pointsLoaded = false;
    this.camerasLoaded = false;
    this.mode = 'none';
    this.primitives = [];
    this.cameras = [];
    this.currentCam = 0;
    this.trainingMode = 'manual';
    
    if (this.trainingToggleBtn) {
      this.trainingToggleBtn.textContent = 'Train';
      this.trainingToggleBtn.style.background = '#44aa44';
    }
    if (this.autoTrainBtn) {
      this.autoTrainBtn.textContent = 'Auto: OFF';
      this.autoTrainBtn.style.background = '#333333';
      this.autoTrainBtn.style.color = '#888';
    }
    if (this.counterDisplay) {
      this.counterDisplay.textContent = '0';
    }
    
    await this.rnd.cleanup();
    await this.rnd.init(document.querySelector("#webgpu-canvas") as HTMLElement);
    
    this.showMainMenu();
    this.updateUIForMode();
    this.drawMessage('LOAD POINTS TO START');
  }

  showMainMenu() {
    this.menuState = 'main';
    const mainMenu = document.getElementById('mainMenu');
    const pointsMenu = document.getElementById('pointsMenu');
    const camerasMenu = document.getElementById('camerasMenu');
    
    if (mainMenu) mainMenu.style.display = 'flex';
    if (pointsMenu) pointsMenu.style.display = 'none';
    if (camerasMenu) camerasMenu.style.display = 'none';
  }

  showPointsMenu() {
    this.menuState = 'points';
    const mainMenu = document.getElementById('mainMenu');
    const pointsMenu = document.getElementById('pointsMenu');
    const camerasMenu = document.getElementById('camerasMenu');
    
    if (mainMenu) mainMenu.style.display = 'none';
    if (pointsMenu) pointsMenu.style.display = 'flex';
    if (camerasMenu) camerasMenu.style.display = 'none';
  }

  showCamerasMenu() {
    this.menuState = 'cameras';
    const mainMenu = document.getElementById('mainMenu');
    const pointsMenu = document.getElementById('pointsMenu');
    const camerasMenu = document.getElementById('camerasMenu');
    
    if (mainMenu) mainMenu.style.display = 'none';
    if (pointsMenu) pointsMenu.style.display = 'none';
    if (camerasMenu) camerasMenu.style.display = 'flex';
  }

  updateUIForMode() {
    const sidebar = document.getElementById('sidebar');
    const trainingControls = document.getElementById('trainingControls');
    const cleanBtn = document.getElementById('menuClean');
    const loadCamerasBtn = document.getElementById('menuLoadCameras');
    
    if (sidebar) {
      sidebar.style.display = (this.pointsLoaded && this.camerasLoaded) ? 'none' : 'flex';
    }
    if (trainingControls) {
      trainingControls.style.display = (this.pointsLoaded && this.camerasLoaded) ? 'flex' : 'none';
    }
    if (cleanBtn) {
      cleanBtn.style.display = (this.pointsLoaded || this.camerasLoaded) ? 'flex' : 'none';
    }
    if (loadCamerasBtn) {
      loadCamerasBtn.style.display = this.pointsLoaded ? 'flex' : 'none';
    }
  }

  async loadPointsFile(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    
    this.drawMessage('LOADING POINTS...');
    
    try {
      const extension = file.name.split('.').pop()?.toLowerCase();
      
      if (extension === 'json') {
        await this.loadPointsFromJSON(file);
      } else if (extension === 'bin') {
        await this.loadPointsFromBinary(file);
      } else {
        throw new Error('Unsupported file format');
      }
      
      this.pointsLoaded = true;
      this.isReady = true;
      this.mode = this.camerasLoaded ? 'training' : 'viewer';
      
      if (this.mode === 'viewer') {
        this.rnd.attachToDraw(this.primitives);
      }
      
      this.updateUIForMode();
      this.showMainMenu();
    } catch (error) {
      console.error("Failed to load points:", error);
      this.drawMessage('ERROR LOADING POINTS\nTRY AGAIN');
    }
  }

  async loadPointsFromJSON(file: File) {
    const text = await file.text();
    const gaussiansData: GaussianData[] = JSON.parse(text);
    
    this.primitives = [];
    for (let i = 0; i < gaussiansData.length; i++) {
      const g = gaussiansData[i];
      this.primitives.push(new primitive(
        new vec3(g.pos[0], g.pos[1], g.pos[2]),
        new vec3(0.02),
        new vec4(0, 0, 0, 1),
        0.6,
        new vec4(g.color[0], g.color[1], g.color[2], g.color[3]),
      ));
      this.primitives[i].index = i;
    }
  }

  async loadPointsFromBinary(file: File) {
    const arrayBuffer = await file.arrayBuffer();
    const floatData = new Float32Array(arrayBuffer);
    
    const stride = 14;
    const numPrimitives = floatData.length / stride;
    
    this.primitives = [];
    for (let i = 0; i < numPrimitives; i++) {
      const offset = i * stride;
      
      const s0 = floatData[offset + 3];
      const s1 = floatData[offset + 4];
      const s2 = floatData[offset + 5];

      this.primitives.push(new primitive(
        new vec3(floatData[offset + 0], floatData[offset + 1], floatData[offset + 2]),
        new vec3(
          s0 > 1.0 ? 1.0 : s0,
          s1 > 1.0 ? 1.0 : s1,
          s2 > 1.0 ? 1.0 : s2
        ),
        new vec4(
          floatData[offset + 6],
          floatData[offset + 7],
          floatData[offset + 8],
          floatData[offset + 9]
        ),
        floatData[offset + 10],
        new vec4(
          floatData[offset + 11],
          floatData[offset + 12],
          floatData[offset + 13],
          1.0
        ),
      ));
    }
  }

  async loadCamerasFile(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    
    try {
      const text = await file.text();
      this.cameras = JSON.parse(text);
      this.camerasLoaded = true;
      this.currentCam = 0;
      
      if (this.pointsLoaded) {
        this.mode = 'training';
        this.rnd.attachToDraw(this.primitives);
        await this.loadCurrentImage();
        this.updateCameraDisplay();
      }
      
      this.updateUIForMode();
      this.showMainMenu();
    } catch (error) {
      console.error("Failed to load cameras:", error);
    }
  }

  async loadCurrentImage() {
    if (!this.camerasLoaded || this.cameras.length === 0) return;
    
    const targetW = this.rnd.canvas.width;
    const targetH = this.rnd.canvas.height;
    const cam = this.cameras[this.currentCam];
    const Path = `bin/images/${cam.name}`;
    
    try {
      const response = await fetch(Path);
      if (!response.ok) {
        console.error(`Failed to load image: ${Path}`);
        return;
      }
      
      const blob = await response.blob();
      if (blob.size === 0) return;

      const imageBitmap = await createImageBitmap(blob, {
        colorSpaceConversion: 'default',
        resizeWidth: targetW,
        resizeHeight: targetH,
        resizeQuality: 'high'
      });

      this.rnd.device.queue.copyExternalImageToTexture(
        { source: imageBitmap },
        { texture: this.rnd.imageTexture },
        [imageBitmap.width, imageBitmap.height]
      );
      
      this.rnd.imageTextureView = this.rnd.imageTexture.createView();
      this.rnd.reload();
    } catch (error) {
      console.error(`Error loading image ${Path}:`, error);
    }
  }

  prevCamera() {
    if (!this.camerasLoaded || this.cameras.length === 0) return;
    this.currentCam--;
    if (this.currentCam < 0) this.currentCam = this.cameras.length - 1;
    this.updateCameraDisplay();
    this.loadCurrentImage();
  }

  nextCamera() {
    if (!this.camerasLoaded || this.cameras.length === 0) return;
    this.currentCam++;
    if (this.currentCam > this.cameras.length - 1) this.currentCam = 0;
    this.updateCameraDisplay();
    this.loadCurrentImage();
  }

  toggleTraining() {
    this.isTrainingActive = !this.isTrainingActive;
    if (this.trainingToggleBtn) {
      if (this.isTrainingActive) {
        this.trainingToggleBtn.textContent = 'Stop';
        this.trainingToggleBtn.style.background = '#ff4444';
      } else {
        this.trainingToggleBtn.textContent = 'Train';
        this.trainingToggleBtn.style.background = '#44aa44';
      }
    }
  }

  toggleAutoTraining() {
    this.trainingMode = this.trainingMode === 'auto' ? 'manual' : 'auto';
    if (this.autoTrainBtn) {
      if (this.trainingMode === 'auto') {
        this.autoTrainBtn.textContent = 'Auto: ON';
        this.autoTrainBtn.style.background = '#ff8800';
        this.autoTrainBtn.style.color = '#000';
      } else {
        this.autoTrainBtn.textContent = 'Auto: OFF';
        this.autoTrainBtn.style.background = '#333333';
        this.autoTrainBtn.style.color = '#888';
      }
    }
  }

  updateCameraDisplay() {
    if (this.counterDisplay) {
      this.counterDisplay.textContent = this.currentCam.toString();
    }
  }

  async startRenderLoop() {
    let iter = 0;
    
    const draw = async () => {
      if (this.isReady) {
        await this.rnd.start();

        if (this.mode === 'training' && this.isTrainingActive && this.cameras.length > 0) {
          if (this.trainingMode === 'auto' && iter % 100 === 0) {
            this.currentCam = Math.floor(Math.random() * this.cameras.length);
            this.updateCameraDisplay();
            this.loadCurrentImage();
          }
          
          const cam = this.cameras[this.currentCam];
          const loc = new vec3(cam.loc[0], cam.loc[1], cam.loc[2]);
          const at = new vec3(cam.at[0], cam.at[1], cam.at[2]);
          const up = new vec3(cam.up[0], cam.up[1], cam.up[2]);
          const target = at.add(loc);
          this.rnd.controls.cam.set(loc, target, up);
        }

        await this.rnd.end(this.isTrainingActive && this.mode === 'training');
      }
      
      iter++;
      window.requestAnimationFrame(draw);
    };
    
    draw();
  }
}

const init = async () => {
  const viewer = new GaussianViewer();
  await viewer.init();
};

init();