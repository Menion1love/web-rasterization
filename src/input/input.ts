/***************************************************************
 * Copyright (C) 2026
 *    Computer Graphics Support Group of 30 Phys-Math Lyceum
 ***************************************************************/

/** FILE NAME   : input.ts
 *  PURPOSE     : Gaussian splatting project.
 *  PROGRAMMER  : CGSG'An'2026.
 *                Timofey Hudyakov (TH4).
 *  LAST UPDATE : 09.06.2026
 * 
 *  No part of this file may be changed without agreement of
 *  Computer Graphics Support Group of 30 Phys-Math Lyceum
 */

/** 
 * * Input class 
 **/
class _input {
  /** #private parameters */
  // Keyboard state tracking
  private keysPressed = new Set<string>();      // Currently held keys
  private keysJustPressed = new Set<string>();  // Keys pressed this frame
  
  // Mouse state tracking (0=left, 1=middle, 2=right)
  private mousePressed = [false, false, false];     // Currently held buttons
  private mouseJustPressed = [false, false, false]; // Buttons pressed this frame

  // Accumulated mouse movement for smooth input
  private accumulatedDX: number = 0;
  private accumulatedDY: number = 0;
  private accumulatedDZ: number = 0;

  // HTML elements 
  private canvasID: HTMLElement;
  
  /** #public parameters */
  // Mouse position and movement
  public mouseX: number = 0;        // Current mouse X position
  public mouseY: number = 0;        // Current mouse Y position
  public mouseDX: number = 0;       // Mouse X delta (movement)
  public mouseDY: number = 0;       // Mouse Y delta (movement)
  public mouseDZ: number = 0;       // Mouse wheel delta

  /**
   * Input class constructor function.
   * @param None.
   * @returns None.
   */
  public constructor() {
    this.canvasID = document.querySelector("#app")!;
    
    // Track mouse over canvas
    this.canvasID.addEventListener("mouseenter", () => {
    });
    
    this.canvasID.addEventListener("mouseleave", () => {
      this.mousePressed.fill(false);
      this.accumulatedDX = this.accumulatedDY = 0;
    });
    
    // Keyboard events
    document.addEventListener("keydown", (e: KeyboardEvent) => {
      if (!this.keysPressed.has(e.code)) {
        this.keysJustPressed.add(e.code);  // Mark as just pressed
        this.keysPressed.add(e.code);      // Mark as held
      }
    });
    
    document.addEventListener("keyup", (e: KeyboardEvent) => {
      this.keysPressed.delete(e.code);     // Remove from held keys
    });
    
    // Mouse button events - only on canvas
    this.canvasID.addEventListener("mousedown", (e: MouseEvent) => {
      if (!this.mousePressed[e.button]) {
        this.mouseJustPressed[e.button] = true;  // Mark as just pressed
        this.mousePressed[e.button] = true;      // Mark as held
      }
      this.mouseX = e.clientX;
      this.mouseY = e.clientY;
    });
    
    this.canvasID.addEventListener("mouseup", (e: MouseEvent) => {
      this.mousePressed[e.button] = false;       // Remove from held buttons
    });
    
    // Mouse movement - only on canvas
    this.canvasID.addEventListener("mousemove", (e: MouseEvent) => {
      const deltaX = e.clientX - this.mouseX;    // Calculate delta
      const deltaY = e.clientY - this.mouseY;
      
      this.accumulatedDX += deltaX;              // Accumulate movement
      this.accumulatedDY += deltaY;
      
      this.mouseX = e.clientX;                   // Update position
      this.mouseY = e.clientY;
    });
    
    // Mouse wheel - prevent page zoom everywhere
    document.addEventListener("wheel", (e: WheelEvent) => {
      if (e.ctrlKey) {
        e.preventDefault(); // Prevent zoom
      }
    }, { passive: false });
    
    this.canvasID.addEventListener("wheel", (e: WheelEvent) => {
      e.preventDefault();
      this.accumulatedDZ += e.deltaY;
    });
    
    // Prevent context menu
    this.canvasID.addEventListener("contextmenu", (e: Event) => e.preventDefault());
  } /** End of 'constructor' function */

  /**
   * Check if key is currently held down function.
   * @param key Key code to check.
   * @returns True if key is pressed.
   */
  public isKeyPressed(key: string): boolean { return this.keysPressed.has(key); }
  
  /**
   * Check if key was just pressed this frame function.
   * @param key Key code to check.
   * @returns True if key was just pressed.
   */
  public isKeyJustPressed(key: string): boolean { return this.keysJustPressed.has(key); }
  
  /**
   * Check if mouse button is currently held down function.
   * @param btn Mouse button index (0=left, 1=middle, 2=right).
   * @returns True if button is pressed.
   */
  public isMousePressed(btn = 0): boolean { return this.mousePressed[btn]; }
  
  /**
   * Check if mouse button was just pressed this frame function.
   * @param btn Mouse button index (0=left, 1=middle, 2=right).
   * @returns True if button was just pressed.
   */
  public isMouseJustPressed(btn = 0): boolean { return this.mouseJustPressed[btn]; }
  
  // Convenience getters
  public get leftClick(): boolean { return this.mousePressed[0]; }
  public get rightClick(): boolean { return this.mousePressed[2]; }
  public get leftClickJust(): boolean { return this.mouseJustPressed[0]; }
  
  /**
   * Input responce function.
   * @param None.
   * @returns None.
   */
  public async response(): Promise<void> {
    
    // Transfer accumulated movement to current deltas
    this.mouseDX = this.accumulatedDX;
    this.mouseDY = this.accumulatedDY;
    this.mouseDZ = this.accumulatedDZ;
    
    // Reset accumulated movement
    this.accumulatedDX = this.accumulatedDY = this.accumulatedDZ = 0;
  } /** End of 'response' function */
  
  /**
   * Clear input state function.
   * @param None.
   * @returns None.
   */
  public async clear(): Promise<void> {
    this.keysJustPressed.clear();           // Clear just pressed keys
    this.mouseJustPressed.fill(false);     // Clear just pressed buttons
  }

} /* End of '_input' class */

/** Input variable */
const input: _input = new _input();

/** EXPORTS */
export { input };

/** END OF 'input.ts' FILE */