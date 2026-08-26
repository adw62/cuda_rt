import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

// WASD + mouselook free camera, pointer-locked to the canvas. This directly
// drives the camera object passed in -- there is no separate "editor" camera,
// per the design: the viewport camera you fly IS the render camera.
export class FlyControls {
  constructor(camera, domElement) {
    this.camera = camera;
    this.domElement = domElement;
    this.controls = new PointerLockControls(camera, domElement);
    this.speed = 2.5; // world units / second
    this.boost = 3.0;
    this.keys = new Set();
    this.enabled = false;

    this._onKeyDown = (e) => this.keys.add(e.code);
    this._onKeyUp = (e) => this.keys.delete(e.code);
  }

  get isLocked() {
    return this.controls.isLocked;
  }

  enable(onLocked, onUnlocked) {
    this.enabled = true;
    document.addEventListener('keydown', this._onKeyDown);
    document.addEventListener('keyup', this._onKeyUp);
    this.controls.addEventListener('lock', () => onLocked && onLocked());
    this.controls.addEventListener('unlock', () => onUnlocked && onUnlocked());
    this.controls.lock();
  }

  disable() {
    this.enabled = false;
    this.keys.clear();
    document.removeEventListener('keydown', this._onKeyDown);
    document.removeEventListener('keyup', this._onKeyUp);
    if (this.controls.isLocked) this.controls.unlock();
  }

  update(dt) {
    if (!this.enabled || !this.controls.isLocked) return;
    const speed = this.speed * (this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? this.boost : 1);
    const step = speed * dt;
    if (this.keys.has('KeyW')) this.controls.moveForward(step);
    if (this.keys.has('KeyS')) this.controls.moveForward(-step);
    if (this.keys.has('KeyD')) this.controls.moveRight(step);
    if (this.keys.has('KeyA')) this.controls.moveRight(-step);
    if (this.keys.has('KeyE') || this.keys.has('Space')) this.camera.position.y += step;
    if (this.keys.has('KeyQ') || this.keys.has('ControlLeft')) this.camera.position.y -= step;
  }

  // World-space forward/up for the current camera orientation.
  getBasis() {
    const forward = new THREE.Vector3();
    this.camera.getWorldDirection(forward);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    return { forward: forward.toArray(), up: up.toArray() };
  }
}
