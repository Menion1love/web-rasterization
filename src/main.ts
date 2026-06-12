import './style.css'
import { render, primitive } from "./render.ts"
import { timer } from "./input/timer.ts"
import { vec3 } from "./mth/mth_vec3.ts"
import { vec4 } from "./mth/mth_vec4.ts"


const init = async () => {
  let canvasId = document.querySelector("#webgpu-canvas");
  let rnd = new render();
  await rnd.init(canvasId as HTMLElement);
  let primitives: primitive[] = [];
  let js = await fetch("./bin/points.json");
  let text = await js.json();

  for (let i = 0; i < text.length; i++) {
    primitives.push(new primitive(
      new vec3(text[i].pos[0], -text[i].pos[1], text[i].pos[2]),
      new vec3(text[i].scale, text[i].scale, text[i].scale),
      new vec4(0, 0, 0, 1),
      0.6,
      new vec4(text[i].color[0], text[i].color[1], text[i].color[2], text[i].color[3]),
    ));
    primitives[i].index = i;
  }
    // primitives.push(new primitive(
    //     new vec3(0),
    //     new vec3(0.8),
    //     new vec4(0, 0, 0, 1),
    //     0.8,
    //     new vec4(1, 0, 1, 1),
    //   ));
    //     primitives[0].index = 0;

    // primitives.push(new primitive(
    //     new vec3(1),
    //     new vec3(0.8),
    //     new vec4(0, 0, 0, 1),
    //     0.9,
    //     new vec4(0, 0, 1, 1),
    //   ));
    // primitives[1].index = 1;
    // primitives.push(new primitive(
    //     new vec3(-1, 0, 0),
    //     new vec3(0.8),
    //     new vec4(0, 0, 0, 1),
    //     0.4,
    //     new vec4(0, 1, 0, 1),
    //   ));
    // primitives[2].index = 2;

  const draw = async () => {
    rnd.start();

    // primitives[1].position = new vec3(4, Math.sin(timer.time) * 2, 0);
    // primitives[2].position = new vec3(-4, Math.cos(timer.time) * 2, 0);

    for (let i = 0; i < primitives.length; i++) {
      rnd.draw(primitives[i]);
    }

    await rnd.end();  
    window.requestAnimationFrame(draw);
  };
  draw();
};

init();
