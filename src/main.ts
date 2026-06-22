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
  let js = await fetch("./bin/train/gaussians.json");
  let jscam = await fetch("./bin/train/cam.json");
  let text = await js.json();
  let cam = await jscam.json();
  let curCam = 0;

  let load = async () => {
    const targetW = rnd.canvas.width;
    const targetH = rnd.canvas.height;

    const Path = `bin/images/${cam[curCam].name}`;
    // const Path = `bin/images/true2.png`;

    const response = await fetch(Path);
    if (!response.ok) {
      console.error(`Failed to load image: ${Path}, status: ${response.status}`);
      return;
    }
    const blob = await response.blob();
    if (blob.size === 0) {
      console.error(`Empty blob for: ${Path}`);
      return;
    }

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
  // const response = await fetch("./bin/scene_data_4.bin");
  // const arrayBuffer = await response.arrayBuffer();
  
  // const floatData = new Float32Array(arrayBuffer);
  
  // const stride = 14; 
  // const numPrimitives = floatData.length / stride;
  
  // for (let i = 0; i < numPrimitives; i++) {
  //   const offset = i * stride;
    
  //   const s0 = floatData[offset + 3];
  //   const s1 = floatData[offset + 4];
  //   const s2 = floatData[offset + 5];

  //   primitives.push(new primitive(
  //     new vec3(floatData[offset + 0], floatData[offset + 1], floatData[offset + 2]),
      
  //     new vec3(
  //       s0 > 1.0 ? 1.0 : s0,
  //       s1 > 1.0 ? 1.0 : s1,
  //       s2 > 1.0 ? 1.0 : s2
  //     ),
      
  //     new vec4(
  //       floatData[offset + 6], // X
  //       floatData[offset + 7], // Y
  //       floatData[offset + 8], // Z
  //       floatData[offset + 9]  // W
  //     ),
      
  //     floatData[offset + 10],
      
  //     new vec4(
  //       floatData[offset + 11], // R
  //       floatData[offset + 12], // G
  //       floatData[offset + 13], // B
  //       1.0                     // A
  //     ),
  //   ));
  // }

  for (let i = 0; i < text.length; i++) {
    primitives.push(new primitive(
      new vec3(text[i].pos[0], text[i].pos[1], text[i].pos[2]),
      new vec3(0.02),
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

    // True2 png params
    // primitives.push(new primitive(
    //     new vec3(0),
    //     new vec3(0.6, 0.2, 0.2),
    //     new vec4(0.0, 0.0, 0.7, 1),
    //     0.8,
    //     new vec4(1, 0, 1, 1),
    //   ));

    // primitives.push(new primitive(
    //     new vec3(1),
    //     new vec3(0.2, 0.8, 0.2),
    //     new vec4(0, 0, 0, 1),
    //     0.9,
    //     new vec4(0, 0, 1.0, 1),
    //   ));

    // primitives.push(new primitive(
    //     new vec3(-1),
    //     new vec3(0.6, 0.2, 0.1),
    //     new vec4(0.0, 0.0, 0.7, 1),
    //     0.8,
    //     new vec4(1.0, 0, 1, 1),
    //   ));

    // primitives.push(new primitive(
    //     new vec3(1),
    //     new vec3(0.2, 0.8, 0.2),
    //     new vec4(0, 0, 0.0, 1),
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

  let flag = true;

  renderButton?.addEventListener('click', async (): Promise<void> => {
    flag = !flag;
  });

  rnd.attachToDraw(primitives);

  let iter = 0;
  
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
    // if (iter % 100 == 0) {
    // {
    //   curCam = Math.floor(Math.random() * cam.length);
    //   if (curCam >= cam.length)
    //     curCam = cam.length - 1;
    //   await load();
    // }


    if (flag && cam.length > 0)
    {
      let loc = new vec3(cam[curCam].loc[0], cam[curCam].loc[1], cam[curCam].loc[2]);
      let at = new vec3(cam[curCam].at[0], cam[curCam].at[1], cam[curCam].at[2]);
      let up = new vec3(cam[curCam].up[0], cam[curCam].up[1], cam[curCam].up[2]);
      let target = at.add(loc);
      rnd.controls.cam.set(loc, target, up);
    }

    await rnd.end(flag);  
    iter++;
    // if (iter > 4500)
    // {
    //   iter = 0;
    //   rnd.IterationsCount = 1;
    //   curCam++;
    // }
    //rnd.primsClear();
    window.requestAnimationFrame(draw);
  };
  draw();
};

init();
