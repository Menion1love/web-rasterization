import './style.css'
import { render, primitive } from "./render.ts"
import { timer } from "./input/timer.ts"
import { input } from "./input/input.ts"
import { vec3 } from "./mth/mth_vec3.ts"
import { vec4 } from "./mth/mth_vec4.ts"

/** Convert degrees to radians */
const d2R = (rad: number): number => rad * Math.PI / 180;

const init = async () => {
  let canvasId = document.querySelector("#webgpu-canvas");
  let rnd = new render();
  await rnd.init(canvasId as HTMLElement);
  let primitives: primitive[] = [];
  let js = await fetch("./bin/gaussians.json");
  let jscam = await fetch("./bin/cam1.json");
  let text = await js.json();
  let cam = await jscam.json();
  let curCam = 0;

  let load = async () => {
    const targetW = rnd.canvas.width;
    const targetH = rnd.canvas.height;

    const Path = `bin/images/${cam[curCam].name}`;
    // const Path = `bin/images/true.png`;

    const response = await fetch(Path);
    const blob = await response.blob();

    const imageBitmap = await createImageBitmap(blob, {
      colorSpaceConversion: 'default',
    
      resizeWidth: targetW,
      resizeHeight: targetH,
      resizeQuality: 'high' 
    });

    rnd.device.queue.copyExternalImageToTexture(
      { source: imageBitmap }, 
      { texture: rnd.imageTexture },
      [imageBitmap.width, imageBitmap.height] 
    );
    
    rnd.imageTextureView = rnd.imageTexture.createView();
    rnd.reload();

  }

  for (let i = 0; i < text.length; i++) {
    primitives.push(new primitive(
      new vec3(text[i].pos[0], text[i].pos[1], text[i].pos[2]),
      new vec3(0.1),
      new vec4(0, 0, 0, 1),
      0.6,
      new vec4(text[i].color[0], text[i].color[1], text[i].color[2], text[i].color[3]),
    ));
    primitives[i].index = i;
  }
  await load();



    // primitives.push(new primitive(
    //     new vec3(0),
    //     new vec3(0.4, 0.2, 0.8),
    //     new vec4(0, 0, 0, 1),
    //     0.8,
    //     new vec4(1, 0, 1, 1),
    //   ));

    // primitives.push(new primitive(
    //     new vec3(1),
    //     new vec3(0.2, 0.5, 0.2),
    //     new vec4(0, 0, 0, 1),
    //     0.9,
    //     new vec4(0, 0, 1, 1),
    //   ));
  
    // True1 png params
    // primitives.push(new primitive(
    //     new vec3(0),
    //     new vec3(0.8, 0.6, 0.4),
    //     new vec4(0.0, 0.0, 0.6, 1),
    //     0.8,
    //     new vec4(1, 0, 1, 1),
    //   ));

    // primitives.push(new primitive(
    //     new vec3(0),
    //     new vec3(0.2, 0.2, 0.01),
    //     new vec4(0.0, 0.0, 0.0, 1),
    //     0.8,
    //     new vec4(1, 0, 1, 1),
    //   ));

    // primitives.push(new primitive(
    //     new vec3(1),
    //     new vec3(0.2, 0.2, 0.2),
    //     new vec4(0, 0, 0, 1),
    //     0.9,
    //     new vec4(0, 0, 1.0, 1),
    //   ));

  const prevButton = document.getElementById('prevBtn') as HTMLButtonElement;
  const nextButton = document.getElementById('nextBtn') as HTMLButtonElement;
  const valueDisplay = document.getElementById('counterValue') as HTMLSpanElement;
  const renderButton = document.getElementById('renderBtn') as HTMLButtonElement;

  let updateDisplay = function(): void {
    if (valueDisplay) {
      valueDisplay.textContent = curCam.toString();
    }
  }

  prevButton?.addEventListener('click', async (): Promise<void> => {
    curCam--;
    if (curCam < 0)
      curCam = cam.length - 1;
    updateDisplay();
    load();
  });

  nextButton?.addEventListener('click', async (): Promise<void> => {
    curCam++;
    if (curCam > cam.length - 1)
      curCam = 0;
    updateDisplay();     
    load();
  });

  let flag = false;


  renderButton?.addEventListener('click', async (): Promise<void> => {
    flag = true;
  });

  rnd.attachToDraw(primitives);
  
  const draw = async () => {
    rnd.start();
    // primitives[0].rotation = new vec4(Math.sin(timer.time), 0, 0, 1);

    // primitives.push(new primitive(
    //     new vec3(1),
    //     new vec3(0.2, 0.2, 0.2),
    //     new vec4(0, 0, 0, 1),
    //     0.9,
    //     new vec4(0, 0, 1.0, 1),
    //   ));


    if (cam.length > 0)
    {
      let loc = new vec3(cam[curCam].loc[0], cam[curCam].loc[1], cam[curCam].loc[2]);
      let at = new vec3(cam[curCam].at[0], cam[curCam].at[1], cam[curCam].at[2]);
      let up = new vec3(cam[curCam].up[0], cam[curCam].up[1], cam[curCam].up[2]);
      let target = at.add(loc);
      rnd.controls.cam.set(loc, target, up);
    }

    await rnd.end(true);  
    flag = false;
    //rnd.primsClear();
    window.requestAnimationFrame(draw);
  };
  draw();
};

init();
