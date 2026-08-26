import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SceneModel } from './scene-model.js';
import { FlyControls } from './fly-controls.js';
import { Gizmo } from './gizmo.js';
import { Outliner } from './outliner.js';
import { Inspector } from './inspector.js';
import { Timeline } from './timeline.js';
import { bakePhysics } from './physics.js';
import {
  sceneToJSON,
  sceneFromJSON,
  requestRender,
  pollRenderStatus,
  saveSceneToServer,
  loadSceneFromServer,
  listScenes,
} from './export.js';

const model = new SceneModel();
model.addObject('sphere', [0, 0, -1], { size: 0.4, diffusion: [0.7, 0, 0], ambient: [0.1, 0, 0] });

let currentFrame = 0;
let selectedId = null;
let isFlying = false;
let isPlaying = false;
let lastPlayTime = 0;
const PLAYBACK_FPS = 24;

// --- Three.js scene setup -------------------------------------------------

const viewportEl = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
viewportEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1b1f);
scene.add(new THREE.GridHelper(10, 20, 0x444444, 0x2a2a2a));
scene.add(new THREE.AmbientLight(0xffffff, 0.25));

const viewportCamera = new THREE.PerspectiveCamera(60, 1, 0.05, 5000);
viewportCamera.position.set(1.5, 1.2, 3.5);

const orbitControls = new OrbitControls(viewportCamera, renderer.domElement);
orbitControls.target.set(0, 0, -1);

const flyControls = new FlyControls(viewportCamera, renderer.domElement);

const gizmo = new Gizmo(viewportCamera, renderer.domElement, scene, orbitControls);

// Point light + small marker sphere standing in for the raytraced point light.
const pointLight = new THREE.PointLight(0xffffff, 1.2, 0, 0.2);
scene.add(pointLight);
const lightHelper = new THREE.Mesh(
  new THREE.SphereGeometry(0.08, 12, 8),
  new THREE.MeshBasicMaterial({ color: 0xffee88 })
);
scene.add(lightHelper);

// Camera frustum visualization for the render camera's current sampled pose.
const dummyCam = new THREE.PerspectiveCamera(50, 4 / 3, 0.05, 1);
scene.add(dummyCam);
const cameraHelper = new THREE.CameraHelper(dummyCam);
scene.add(cameraHelper);

const meshes = new Map(); // objectId -> THREE.Mesh

// Cheap stand-in for the CUDA octant checker (not a pixel-accurate match --
// this is a UV-mapped grid, the renderer's is an axis-sign octant split) --
// good enough to see a checkered sphere actually spinning in the viewport.
function makeCheckerTexture() {
  const size = 128;
  const squares = 8;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const step = size / squares;
  for (let y = 0; y < squares; y++) {
    for (let x = 0; x < squares; x++) {
      ctx.fillStyle = (x + y) % 2 === 0 ? '#ffffff' : '#555555';
      ctx.fillRect(x * step, y * step, step, step);
    }
  }
  return new THREE.CanvasTexture(canvas);
}
const checkerTexture = makeCheckerTexture();

// Renderer-side geometry only ever has one of: sphere radius, box half-extents,
// or plane normal, matching the `type`-dispatched param1 on the CUDA side.
function geometryFor(obj) {
  if (obj.type === 'box') {
    const [hx, hy, hz] = obj.halfExtents;
    return new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2);
  }
  if (obj.type === 'plane') {
    return new THREE.PlaneGeometry(20, 20); // finite visual stand-in for the renderer's infinite plane
  }
  return new THREE.SphereGeometry(Math.max(obj.size, 0.001), 24, 16);
}

function orientMesh(mesh, obj) {
  if (obj.type === 'plane') {
    const n = new THREE.Vector3(...obj.normal).normalize();
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
  } else if (obj.type === 'box') {
    mesh.quaternion.set(...obj.rotation);
  } else {
    mesh.quaternion.identity();
  }
}

function applyCheckerMap(mat, obj) {
  mat.map = obj.type === 'sphere' && obj.checker ? checkerTexture : null;
  mat.needsUpdate = true;
}

// Flat opacity, no bending -- a real-time refraction preview isn't worth the
// complexity here (same trade-off as the checker: the viewport gives you a
// visual cue, the CUDA render is the actual answer). Opacity is eased so a
// low transparency still reads as visibly "glassy" rather than merely faded.
function applyTransparency(mat, obj) {
  const t = obj.transparency || 0;
  mat.transparent = t > 0;
  mat.opacity = 1 - t * 0.85;
}

function createMeshFor(obj) {
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(...obj.diffusion),
    roughness: 0.6,
    side: obj.type === 'plane' ? THREE.DoubleSide : THREE.FrontSide,
  });
  applyCheckerMap(mat, obj);
  applyTransparency(mat, obj);
  const mesh = new THREE.Mesh(geometryFor(obj), mat);
  orientMesh(mesh, obj);
  scene.add(mesh);
  meshes.set(obj.id, mesh);
  return mesh;
}

function syncObjectVisual(obj) {
  const mesh = meshes.get(obj.id);
  if (!mesh) return;
  mesh.geometry.dispose();
  mesh.geometry = geometryFor(obj);
  mesh.material.color.setRGB(...obj.diffusion);
  applyCheckerMap(mesh.material, obj);
  applyTransparency(mesh.material, obj);
  orientMesh(mesh, obj);
}

function rebuildMeshes() {
  for (const mesh of meshes.values()) scene.remove(mesh);
  meshes.clear();
  for (const obj of model.objects) createMeshFor(obj);
}

for (const obj of model.objects) createMeshFor(obj);

// --- Preview update (this is what must match the CUDA render) ------------

function updatePreview(frame) {
  currentFrame = frame;

  for (const obj of model.objects) {
    const mesh = meshes.get(obj.id);
    if (!mesh) continue;
    mesh.position.set(...model.sampleObjectPosition(obj, frame));
    if (obj.type === 'sphere') {
      mesh.quaternion.set(...model.sampleObjectRotation(obj, frame));
    }
  }

  const lightPos = model.sampleLight(frame);
  lightHelper.position.set(...lightPos);
  pointLight.position.copy(lightHelper.position);

  if (!isFlying) {
    const cam = model.sampleCamera(frame);
    dummyCam.position.set(...cam.position);
    dummyCam.up.set(...cam.up);
    dummyCam.lookAt(
      cam.position[0] + cam.forward[0],
      cam.position[1] + cam.forward[1],
      cam.position[2] + cam.forward[2]
    );
    dummyCam.updateMatrixWorld(true);
    cameraHelper.update();
  }

  timeline.setFrame(frame);
  timeline.render();
}

// --- Outliner / Inspector --------------------------------------------------

const outlinerEl = document.getElementById('outliner-list');
const inspectorEl = document.getElementById('inspector');
const camBtn = document.getElementById('select-camera');
const lightBtn = document.getElementById('select-light');

const outliner = new Outliner(outlinerEl, {
  onSelect: (id) => selectEntity(id),
  onAdd: (type) => {
    const obj = model.addObject(type, [0, 0, 0]);
    createMeshFor(obj);
    outliner.render(model.objects);
    selectEntity(obj.id);
    timeline.render();
  },
  onDelete: (id) => {
    const mesh = meshes.get(id);
    if (mesh) {
      scene.remove(mesh);
      meshes.delete(id);
    }
    model.removeObject(id);
    outliner.render(model.objects);
    if (selectedId === id) selectEntity(null);
    timeline.render();
  },
});

const inspector = new Inspector(inspectorEl, {
  onChange: (obj) => syncObjectVisual(obj),
});

function setFixedSelectionUI(id) {
  camBtn.classList.toggle('selected', id === 'camera');
  lightBtn.classList.toggle('selected', id === 'light');
}

// Rotation and scale are static per-object properties (like size/normal
// already were) -- the gizmo is just a nicer way to set them than typing
// numbers into the inspector, so these handlers write straight into the
// object and rebuild its mesh, with no keyframe involved.
function buildGizmoHandlers(obj) {
  const handlers = {
    translate: (pos) => {
      model.insertObjectKeyframe(obj, currentFrame, pos);
      timeline.render();
    },
  };

  if (obj.type === 'box') {
    handlers.rotate = (quat) => {
      obj.rotation = quat;
    };
    handlers.scale = (scaleArr) => {
      obj.halfExtents = obj.halfExtents.map((h, i) => h * scaleArr[i]);
      syncObjectVisual(obj);
      inspector.render(obj);
    };
  } else if (obj.type === 'sphere') {
    // rotation has no visual/render effect on a sphere -- omit that handler
    handlers.scale = (scaleArr) => {
      const uniform = (scaleArr[0] + scaleArr[1] + scaleArr[2]) / 3;
      obj.size = obj.size * uniform;
      syncObjectVisual(obj);
      inspector.render(obj);
    };
  } else if (obj.type === 'plane') {
    handlers.rotate = (quat) => {
      const n = new THREE.Vector3(0, 0, 1).applyQuaternion(new THREE.Quaternion(...quat));
      obj.normal = n.toArray();
      inspector.render(obj);
    };
    // an infinite plane has no real size to scale -- omit that handler
  }

  return handlers;
}

function selectEntity(id) {
  selectedId = id;
  setFixedSelectionUI(id);
  outliner.select(id === 'camera' || id === 'light' ? null : id);
  outliner.render(model.objects);

  if (id === 'camera') {
    inspector.el.innerHTML =
      '<div class="inspector-note">The render camera is set by flying, not by dragging. ' +
      'Press "Fly" then move with WASD + mouse, then "Insert Camera Keyframe" (K) to record a pose.</div>';
    gizmo.detach();
  } else if (id === 'light') {
    inspector.render(null);
    gizmo.attach(lightHelper, {
      translate: (pos) => {
        model.insertLightKeyframe(currentFrame, pos);
        timeline.render();
      },
    });
  } else if (id) {
    const obj = model.getObject(id);
    inspector.render(obj);
    const mesh = meshes.get(id);
    gizmo.attach(mesh, buildGizmoHandlers(obj));
  } else {
    inspector.render(null);
    gizmo.detach();
  }
}

camBtn.onclick = () => selectEntity('camera');
lightBtn.onclick = () => selectEntity('light');

outliner.render(model.objects);
selectEntity(model.objects[0]?.id ?? null);

// --- Timeline ----------------------------------------------------------

const timeline = new Timeline(document.getElementById('timeline'), model, {
  onScrub: (frame) => updatePreview(frame),
  onMoveKeyframe: (trackId, oldFrame, newFrame) => {
    moveKeyframe(trackId, oldFrame, newFrame);
    updatePreview(currentFrame);
  },
  onDeleteKeyframe: (trackId, frame) => {
    deleteKeyframe(trackId, frame);
    updatePreview(currentFrame);
  },
  onAddKeyframe: (trackId, frame) => {
    addKeyframeAtCurrentValue(trackId, frame);
    timeline.render();
  },
});

function findObjOrNull(trackId) {
  return trackId === 'camera' || trackId === 'light' ? null : model.getObject(trackId);
}

function moveKeyframe(trackId, oldFrame, newFrame) {
  if (trackId === 'camera') {
    const kf = model.camera.keyframes.find((k) => k.frame === oldFrame);
    if (kf) model.insertCameraKeyframe(newFrame, { position: kf.position, forward: kf.forward, up: kf.up });
    model.deleteCameraKeyframe(oldFrame);
  } else if (trackId === 'light') {
    const kf = model.light.keyframes.find((k) => k.frame === oldFrame);
    if (kf) model.insertLightKeyframe(newFrame, kf.position);
    model.deleteLightKeyframe(oldFrame);
  } else {
    const obj = findObjOrNull(trackId);
    if (!obj) return;
    const kf = obj.positionKeyframes.find((k) => k.frame === oldFrame);
    if (kf) model.insertObjectKeyframe(obj, newFrame, kf.value);
    model.deleteObjectKeyframe(obj, oldFrame);
  }
}

function deleteKeyframe(trackId, frame) {
  if (trackId === 'camera') model.deleteCameraKeyframe(frame);
  else if (trackId === 'light') model.deleteLightKeyframe(frame);
  else {
    const obj = findObjOrNull(trackId);
    if (obj) model.deleteObjectKeyframe(obj, frame);
  }
}

function addKeyframeAtCurrentValue(trackId, frame) {
  if (trackId === 'camera') {
    const cam = model.sampleCamera(frame);
    model.insertCameraKeyframe(frame, cam);
  } else if (trackId === 'light') {
    model.insertLightKeyframe(frame, model.sampleLight(frame));
  } else {
    const obj = findObjOrNull(trackId);
    if (obj) model.insertObjectKeyframe(obj, frame, model.sampleObjectPosition(obj, frame));
  }
}

// --- Fly mode ------------------------------------------------------------

const flyBtn = document.getElementById('fly-toggle');
const insertCamKeyBtn = document.getElementById('insert-cam-key');

function enterFlyMode() {
  const cam = model.sampleCamera(currentFrame);
  viewportCamera.position.set(...cam.position);
  viewportCamera.up.set(...cam.up);
  viewportCamera.lookAt(
    cam.position[0] + cam.forward[0],
    cam.position[1] + cam.forward[1],
    cam.position[2] + cam.forward[2]
  );
  orbitControls.enabled = false;
  cameraHelper.visible = false;
  isFlying = true;
  flyBtn.textContent = 'Exit Fly (Esc)';
  insertCamKeyBtn.disabled = false;
  flyControls.enable(null, () => exitFlyMode());
}

function exitFlyMode() {
  flyControls.disable();
  isFlying = false;
  flyBtn.textContent = 'Fly Camera';
  insertCamKeyBtn.disabled = true;
  cameraHelper.visible = true;
  const dir = new THREE.Vector3();
  viewportCamera.getWorldDirection(dir);
  orbitControls.target.copy(viewportCamera.position).addScaledVector(dir, 2);
  orbitControls.enabled = true;
  updatePreview(currentFrame);
}

flyBtn.onclick = () => (isFlying ? exitFlyMode() : enterFlyMode());

function insertCameraKeyframeFromFly() {
  if (!isFlying) return;
  const { forward, up } = flyControls.getBasis();
  model.insertCameraKeyframe(currentFrame, {
    position: viewportCamera.position.toArray(),
    forward,
    up,
  });
  timeline.render();
}
insertCamKeyBtn.onclick = insertCameraKeyframeFromFly;

document.addEventListener('keydown', (e) => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.code === 'KeyK' && isFlying) insertCameraKeyframeFromFly();
  if (e.code === 'Digit1') setGizmoMode('translate');
  if (e.code === 'Digit2') setGizmoMode('rotate');
  if (e.code === 'Digit3') setGizmoMode('scale');
});

// --- Gizmo mode (move/rotate/scale) -------------------------------------

const modeButtons = {
  translate: document.getElementById('mode-translate'),
  rotate: document.getElementById('mode-rotate'),
  scale: document.getElementById('mode-scale'),
};

function setGizmoMode(mode) {
  gizmo.setMode(mode);
  for (const [m, btn] of Object.entries(modeButtons)) btn.classList.toggle('selected', m === mode);
}

modeButtons.translate.onclick = () => setGizmoMode('translate');
modeButtons.rotate.onclick = () => setGizmoMode('rotate');
modeButtons.scale.onclick = () => setGizmoMode('scale');
setGizmoMode('translate');

// --- Playback --------------------------------------------------------------

const playBtn = document.getElementById('play-toggle');
playBtn.onclick = () => {
  isPlaying = !isPlaying;
  playBtn.textContent = isPlaying ? 'Pause' : 'Play';
  lastPlayTime = performance.now();
};

// --- Physics -------------------------------------------------------

document.getElementById('bake-physics').onclick = () => {
  const { bakedObjectIds } = bakePhysics(model);
  if (bakedObjectIds.length === 0) {
    alert('Nothing to bake -- enable Gravity on at least one object in the Inspector.');
    return;
  }
  timeline.render();
  updatePreview(currentFrame);
};

// --- Save / Load -------------------------------------------------------

document.getElementById('save-scene').onclick = async () => {
  const name = prompt('Save scene as:', 'my_scene');
  if (!name) return;
  await saveSceneToServer(name, sceneToJSON(model));
};

document.getElementById('load-scene').onclick = async () => {
  const names = await listScenes();
  const name = prompt('Load which scene?\n' + names.join('\n'));
  if (!name) return;
  const data = await loadSceneFromServer(name);
  Object.assign(model, sceneFromJSON(data));
  rebuildMeshes();
  outliner.render(model.objects);
  selectEntity(null);
  updatePreview(0);
  syncResolutionInputs();
};

// --- Render resolution ---------------------------------------------------
// The 3D viewport's own size is unrelated to this -- it only controls the
// pixel dimensions of the actual CUDA render requested via the Render button.

const resXInput = document.getElementById('res-x');
const resYInput = document.getElementById('res-y');

function syncResolutionInputs() {
  resXInput.value = model.resolution.x;
  resYInput.value = model.resolution.y;
}

function setResolution(x, y) {
  model.resolution = { x, y };
  syncResolutionInputs();
}

resXInput.onchange = () => setResolution(Math.max(16, parseInt(resXInput.value, 10) || model.resolution.x), model.resolution.y);
resYInput.onchange = () => setResolution(model.resolution.x, Math.max(16, parseInt(resYInput.value, 10) || model.resolution.y));
document.getElementById('res-preview').onclick = () => setResolution(256, 384);
document.getElementById('res-final').onclick = () => setResolution(1024, 1536);
syncResolutionInputs();

// --- Render ------------------------------------------------------------

const renderBtn = document.getElementById('render-scene');
const renderStatusEl = document.getElementById('render-status');

renderBtn.onclick = async () => {
  renderStatusEl.textContent = 'Submitting...';
  try {
    const { job_id } = await requestRender(sceneToJSON(model));
    poll(job_id);
  } catch (err) {
    renderStatusEl.textContent = 'Failed: ' + err.message;
  }
};

async function poll(jobId) {
  try {
    const status = await pollRenderStatus(jobId);
    renderStatusEl.textContent = `Rendering... ${status.done}/${status.total}`;
    if (status.finished) {
      renderStatusEl.innerHTML = status.error
        ? `Render failed: ${status.error}`
        : `Done: <a href="${status.video_url}" target="_blank">open .avi</a> -- ` +
          `<img src="${status.gif_url}" alt="render preview" style="height:60px;vertical-align:middle;" />`;
      return;
    }
    setTimeout(() => poll(jobId), 1000);
  } catch (err) {
    renderStatusEl.textContent = 'Failed: ' + err.message;
  }
}

// --- Resize + render loop -----------------------------------------------

function resize() {
  const w = viewportEl.clientWidth;
  const h = viewportEl.clientHeight;
  renderer.setSize(w, h);
  viewportCamera.aspect = w / h;
  viewportCamera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

let lastTime = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = (now - lastTime) / 1000;
  lastTime = now;

  flyControls.update(dt);

  if (isPlaying && !isFlying) {
    if (now - lastPlayTime >= 1000 / PLAYBACK_FPS) {
      lastPlayTime = now;
      const next = currentFrame + 1 >= model.frames ? 0 : currentFrame + 1;
      updatePreview(next);
    }
  }

  renderer.render(scene, viewportCamera);
}

updatePreview(0);
timeline.render();
animate();
