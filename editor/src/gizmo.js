import { TransformControls } from 'three/addons/controls/TransformControls.js';

// Wrapper around TransformControls supporting all three modes. Position is
// keyframed (translate); rotation and scale are static per-object properties
// (like a box's size already was) -- the gizmo just gives a nicer way to set
// them than typing numbers into the inspector.
export class Gizmo {
  constructor(camera, domElement, scene, orbitControls) {
    this.controls = new TransformControls(camera, domElement);
    this.controls.setMode('translate');
    this.controls.setSpace('world');
    scene.add(this.controls);

    // Prevent the orbit camera from fighting with a gizmo drag.
    this.controls.addEventListener('dragging-changed', (e) => {
      orbitControls.enabled = !e.value;
    });
  }

  setMode(mode) {
    this.controls.setMode(mode);
  }

  getMode() {
    return this.controls.getMode();
  }

  // handlers: { translate(posArray), rotate(quatArray), scale(scaleArray) }
  // -- any of the three may be omitted if that mode doesn't apply to the
  // attached object's type (e.g. no-op for a sphere's rotation).
  attach(mesh, handlers) {
    this.controls.detach();
    if (!mesh) return;
    this.controls.attach(mesh);
    const handler = () => {
      const mode = this.controls.getMode();
      if (mode === 'translate' && handlers.translate) {
        handlers.translate(mesh.position.toArray());
      } else if (mode === 'rotate' && handlers.rotate) {
        handlers.rotate(mesh.quaternion.toArray());
      } else if (mode === 'scale') {
        if (handlers.scale) handlers.scale(mesh.scale.toArray());
        // scale is baked into the object's real geometry fields by the
        // caller (syncObjectVisual), so the mesh transform itself always
        // resets -- even with no handler, so nothing is left visually
        // distorted (e.g. the light marker, which has no scale concept)
        mesh.scale.set(1, 1, 1);
      }
    };
    this._offListener = () => this.controls.removeEventListener('mouseUp', handler);
    this.controls.addEventListener('mouseUp', handler);
  }

  detach() {
    if (this._offListener) this._offListener();
    this.controls.detach();
  }
}
