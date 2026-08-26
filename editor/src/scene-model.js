// Generic piecewise-linear keyframe sampling. This MUST match scene_io.py's
// sample_track exactly, frame-for-frame, so the live preview here always
// matches what the CUDA renderer will produce for the same scene JSON.

export function lerp(a, b, t) {
  return a.map((v, i) => v + (b[i] - a[i]) * t);
}

export function normalizeVec(v) {
  const mag = Math.hypot(...v);
  return v.map((c) => c / mag);
}

export function sampleTrack(keyframes, frame, { normalize = false } = {}) {
  if (keyframes.length === 0) throw new Error('track has no keyframes');
  const kfs = [...keyframes].sort((a, b) => a.frame - b.frame);
  let value;
  if (kfs.length === 1 || frame <= kfs[0].frame) {
    value = kfs[0].value;
  } else if (frame >= kfs[kfs.length - 1].frame) {
    value = kfs[kfs.length - 1].value;
  } else {
    for (let i = 0; i < kfs.length - 1; i++) {
      const a = kfs[i];
      const b = kfs[i + 1];
      if (frame >= a.frame && frame <= b.frame) {
        const span = b.frame - a.frame;
        const t = span === 0 ? 0 : (frame - a.frame) / span;
        value = lerp(a.value, b.value, t);
        break;
      }
    }
  }
  return normalize ? normalizeVec(value) : value;
}

let nextId = 1;
export function makeId(prefix) {
  return `${prefix}-${nextId++}`;
}

function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function quatMultiply(a, b) {
  // a * b, both (x, y, z, w)
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

function axisAngleQuat(axis, angle) {
  const half = angle / 2;
  const s = Math.sin(half);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(half)];
}

// Rolling-without-slipping quaternion track from a per-frame position track:
// each step rotates about (up x delta) by arclength/radius. Must match
// rolling_quaternions in scene_io.py frame-for-frame so this preview's spin
// agrees with what the CUDA render actually produces.
export function computeRollingQuaternions(positions, radius) {
  const up = [0, 1, 0];
  let quat = [0, 0, 0, 1];
  const out = [quat];
  for (let f = 1; f < positions.length; f++) {
    const prev = positions[f - 1];
    const cur = positions[f];
    const delta = [cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]];
    const dist = Math.hypot(...delta);
    if (dist > 1e-9) {
      const axisRaw = cross3(up, delta);
      const axisMag = Math.hypot(...axisRaw);
      if (axisMag > 1e-9) {
        const axis = axisRaw.map((c) => c / axisMag);
        const dq = axisAngleQuat(axis, dist / radius);
        quat = quatMultiply(dq, quat);
        const mag = Math.hypot(...quat);
        quat = quat.map((c) => c / mag);
      }
    }
    out.push(quat);
  }
  return out;
}

const DEFAULT_MATERIAL = () => ({
  shine: 100,
  reflection: 0.3,
  ambient: [0.1, 0.1, 0.1],
  diffusion: [0.6, 0.6, 0.6],
  specular: [1, 1, 1],
  // 0 = opaque, 1 = fully clear glass; splits into a reflected + refracted
  // ray in rt.cu once > 0, weighted by Fresnel reflectance (ior)
  transparency: 0,
  ior: 1.5,
  gravity: false,
  collision: false,
  initialVelocity: [0, 0, 0],
  velocityRandomization: 0,
});

// Type-specific static (non-keyframed) geometry fields, seeded on creation --
// same pattern as material properties today. `rotation` is a quaternion
// (x,y,z,w); only boxes render with it (planes just store an orientation as
// their `normal`, spheres are rotationally symmetric).
const TYPE_DEFAULTS = {
  sphere: () => ({ size: 0.3, checker: false }),
  box: () => ({ halfExtents: [0.3, 0.3, 0.3], rotation: [0, 0, 0, 1] }),
  plane: () => ({ normal: [0, 1, 0] }),
};

export class SceneModel {
  constructor() {
    this.frames = 100;
    this.resolution = { x: 512, y: 768 };
    this.objects = [];
    this.camera = {
      keyframes: [{ frame: 0, position: [0, 0, 2], forward: [0, 0, -1], up: [0, 1, 0] }],
    };
    this.light = { keyframes: [{ frame: 0, position: [-10, 5, 5] }] };
  }

  addObject(type = 'sphere', position = [0, 0, 0], overrides = {}) {
    if (!TYPE_DEFAULTS[type]) throw new Error(`unknown primitive type: ${type}`);
    const obj = {
      id: makeId(type),
      type,
      ...DEFAULT_MATERIAL(),
      ...TYPE_DEFAULTS[type](),
      ...overrides,
      positionKeyframes: [{ frame: 0, value: position }],
    };
    this.objects.push(obj);
    return obj;
  }

  removeObject(id) {
    this.objects = this.objects.filter((o) => o.id !== id);
  }

  getObject(id) {
    return this.objects.find((o) => o.id === id);
  }

  sampleObjectPosition(obj, frame) {
    return sampleTrack(obj.positionKeyframes, frame);
  }

  // Box rotation is a static gizmo-set pose. A checkered sphere instead spins
  // kinematically (rolling without slipping) from its own position track --
  // recomputed on demand rather than cached, since the position keyframes
  // driving it can change at any time (hand edits, physics bake).
  sampleObjectRotation(obj, frame) {
    if (obj.type === 'box') return obj.rotation;
    if (obj.type === 'sphere' && obj.checker) {
      const positions = Array.from({ length: this.frames }, (_, f) => this.sampleObjectPosition(obj, f));
      const quats = computeRollingQuaternions(positions, obj.size);
      return quats[Math.min(frame, quats.length - 1)];
    }
    return [0, 0, 0, 1];
  }

  sampleCamera(frame) {
    const kfs = this.camera.keyframes;
    return {
      position: sampleTrack(kfs.map((k) => ({ frame: k.frame, value: k.position })), frame),
      forward: sampleTrack(kfs.map((k) => ({ frame: k.frame, value: k.forward })), frame, {
        normalize: true,
      }),
      up: sampleTrack(kfs.map((k) => ({ frame: k.frame, value: k.up })), frame, {
        normalize: true,
      }),
    };
  }

  sampleLight(frame) {
    return sampleTrack(
      this.light.keyframes.map((k) => ({ frame: k.frame, value: k.position })),
      frame
    );
  }

  insertObjectKeyframe(obj, frame, value) {
    const idx = obj.positionKeyframes.findIndex((k) => k.frame === frame);
    if (idx >= 0) obj.positionKeyframes[idx].value = value;
    else obj.positionKeyframes.push({ frame, value });
    obj.positionKeyframes.sort((a, b) => a.frame - b.frame);
  }

  deleteObjectKeyframe(obj, frame) {
    if (obj.positionKeyframes.length <= 1) return;
    obj.positionKeyframes = obj.positionKeyframes.filter((k) => k.frame !== frame);
  }

  insertCameraKeyframe(frame, { position, forward, up }) {
    const idx = this.camera.keyframes.findIndex((k) => k.frame === frame);
    const kf = { frame, position, forward, up };
    if (idx >= 0) this.camera.keyframes[idx] = kf;
    else this.camera.keyframes.push(kf);
    this.camera.keyframes.sort((a, b) => a.frame - b.frame);
  }

  deleteCameraKeyframe(frame) {
    if (this.camera.keyframes.length <= 1) return;
    this.camera.keyframes = this.camera.keyframes.filter((k) => k.frame !== frame);
  }

  insertLightKeyframe(frame, position) {
    const idx = this.light.keyframes.findIndex((k) => k.frame === frame);
    const kf = { frame, position };
    if (idx >= 0) this.light.keyframes[idx] = kf;
    else this.light.keyframes.push(kf);
    this.light.keyframes.sort((a, b) => a.frame - b.frame);
  }

  deleteLightKeyframe(frame) {
    if (this.light.keyframes.length <= 1) return;
    this.light.keyframes = this.light.keyframes.filter((k) => k.frame !== frame);
  }
}
