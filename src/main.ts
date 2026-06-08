import './style.css'
import { render } from "./render.ts"

const init = async () => {
  let canvasId = document.querySelector("#webgpu-canvas");
  let rnd = new render();
  await rnd.init(canvasId);
  
  const draw = async () => {
      rnd.start();  
      rnd.end();  
      window.requestAnimationFrame(draw);
    };
  draw();
};

init();
