import { SceneModel } from './scene-model.js';

export function sceneToJSON(model) {
  return {
    frames: model.frames,
    resolution: model.resolution,
    objects: model.objects.map((o) => ({
      id: o.id,
      type: o.type,
      ...(o.type === 'box' ? { half_extents: o.halfExtents, rotation: o.rotation } : {}),
      ...(o.type === 'plane' ? { normal: o.normal } : {}),
      ...(o.size !== undefined ? { size: o.size } : {}),
      shine: o.shine,
      reflection: o.reflection,
      ambient: o.ambient,
      diffusion: o.diffusion,
      specular: o.specular,
      gravity: !!o.gravity,
      collision: !!o.collision,
      initial_velocity: o.initialVelocity,
      velocity_randomization: o.velocityRandomization,
      position_keyframes: o.positionKeyframes.map((k) => ({ frame: k.frame, pos: k.value })),
    })),
    camera: {
      keyframes: model.camera.keyframes.map((k) => ({
        frame: k.frame,
        position: k.position,
        forward: k.forward,
        up: k.up,
      })),
    },
    light: {
      keyframes: model.light.keyframes.map((k) => ({ frame: k.frame, position: k.position })),
    },
  };
}

export function sceneFromJSON(data) {
  const model = new SceneModel();
  model.frames = data.frames;
  model.resolution = data.resolution;
  model.objects = data.objects.map((o) => ({
    id: o.id,
    type: o.type || 'sphere',
    ...(o.half_extents ? { halfExtents: o.half_extents, rotation: o.rotation || [0, 0, 0, 1] } : {}),
    ...(o.normal ? { normal: o.normal } : {}),
    ...(o.size !== undefined ? { size: o.size } : {}),
    shine: o.shine,
    reflection: o.reflection,
    ambient: o.ambient,
    diffusion: o.diffusion,
    specular: o.specular,
    gravity: !!o.gravity,
    collision: !!o.collision,
    initialVelocity: o.initial_velocity || [0, 0, 0],
    velocityRandomization: o.velocity_randomization || 0,
    positionKeyframes: o.position_keyframes.map((k) => ({ frame: k.frame, value: k.pos })),
  }));
  model.camera = {
    keyframes: data.camera.keyframes.map((k) => ({
      frame: k.frame,
      position: k.position,
      forward: k.forward,
      up: k.up,
    })),
  };
  model.light = {
    keyframes: data.light.keyframes.map((k) => ({ frame: k.frame, position: k.position })),
  };
  return model;
}

async function jsonOrThrow(res) {
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function requestRender(sceneJSON) {
  return fetch('/api/render', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(sceneJSON),
  }).then(jsonOrThrow);
}

export function pollRenderStatus(jobId) {
  return fetch(`/api/render/${jobId}/status`).then(jsonOrThrow);
}

export function saveSceneToServer(name, sceneJSON) {
  return fetch(`/api/scenes/${encodeURIComponent(name)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(sceneJSON),
  }).then(jsonOrThrow);
}

export function loadSceneFromServer(name) {
  return fetch(`/api/scenes/${encodeURIComponent(name)}`).then(jsonOrThrow);
}

export function listScenes() {
  return fetch('/api/scenes').then(jsonOrThrow);
}
