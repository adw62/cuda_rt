import * as CANNON from 'cannon-es';

// Nothing else in the scene has a real-world timescale (frames are just
// discrete keyframe steps), so baking has to pick one -- 24fps is the
// documented assumption for how "gravity" is scaled.
const FIXED_DT = 1 / 24;

function shapeFor(obj) {
  if (obj.type === 'box') {
    const [hx, hy, hz] = obj.halfExtents;
    return new CANNON.Box(new CANNON.Vec3(hx, hy, hz));
  }
  if (obj.type === 'plane') {
    return new CANNON.Plane();
  }
  return new CANNON.Sphere(obj.size);
}

// A cannon Plane's local normal is +Z, same convention already used for the
// Three.js PlaneGeometry orientation in app.js.
function planeOrientation(obj) {
  const q = new CANNON.Quaternion();
  q.setFromVectors(new CANNON.Vec3(0, 0, 1), new CANNON.Vec3(...obj.normal));
  return q;
}

function boxOrientation(obj) {
  const [x, y, z, w] = obj.rotation || [0, 0, 0, 1];
  return new CANNON.Quaternion(x, y, z, w);
}

function randomUnitVector() {
  // uniform point on the unit sphere
  const z = Math.random() * 2 - 1;
  const theta = Math.random() * Math.PI * 2;
  const r = Math.sqrt(1 - z * z);
  return [r * Math.cos(theta), r * Math.sin(theta), z];
}

// Bakes a physics simulation into dense per-frame position keyframes.
// - `collision: true` objects get a collider shape (can be hit / can hit others).
// - `gravity: true` objects are dynamic (mass 1) and have their positionKeyframes
//   replaced wholesale with the simulated result, one keyframe per frame.
// - `gravity: false, collision: true` objects are obstacles (mass 0). If the
//   object already has more than one position keyframe -- i.e. it was hand-
//   animated -- it's a *kinematic* body: its position (and, for correct
//   contact response, velocity) is driven from those existing keyframes every
//   simulated frame, so a moving platform actually interacts with whatever's
//   falling on it instead of being frozen at its frame-0 position. A single-
//   keyframe object is a plain static obstacle, as before.
// - Objects with both flags false are left completely alone.
export function bakePhysics(model, { gravityY = -9.8 } = {}) {
  const world = new CANNON.World();
  world.gravity.set(0, gravityY, 0);

  const dynamicEntries = [];
  const kinematicEntries = [];

  for (const obj of model.objects) {
    if (!obj.gravity && !obj.collision) continue;

    const startPos = model.sampleObjectPosition(obj, 0);
    const isKinematic = !obj.gravity && obj.collision && obj.positionKeyframes.length > 1;

    const body = new CANNON.Body({
      mass: obj.gravity ? 1 : 0,
      position: new CANNON.Vec3(...startPos),
      type: isKinematic ? CANNON.Body.KINEMATIC : undefined,
    });

    if (obj.collision) {
      const shape = shapeFor(obj);
      if (obj.type === 'plane') {
        body.addShape(shape, new CANNON.Vec3(), planeOrientation(obj));
      } else if (obj.type === 'box') {
        body.addShape(shape, new CANNON.Vec3(), boxOrientation(obj));
      } else {
        body.addShape(shape);
      }
    }

    if (obj.gravity) {
      const velocity = [...(obj.initialVelocity || [0, 0, 0])];
      const jitterMag = obj.velocityRandomization || 0;
      if (jitterMag > 0) {
        const jitter = randomUnitVector();
        for (let i = 0; i < 3; i++) velocity[i] += jitter[i] * jitterMag;
      }
      body.velocity.set(...velocity);
    }

    world.addBody(body);

    if (obj.gravity) {
      dynamicEntries.push({ obj, body, positions: [{ frame: 0, value: startPos }] });
    } else if (isKinematic) {
      kinematicEntries.push({ obj, body });
    }
  }

  for (let frame = 1; frame < model.frames; frame++) {
    for (const entry of kinematicEntries) {
      const prev = entry.body.position;
      const next = model.sampleObjectPosition(entry.obj, frame);
      entry.body.velocity.set(
        (next[0] - prev.x) / FIXED_DT,
        (next[1] - prev.y) / FIXED_DT,
        (next[2] - prev.z) / FIXED_DT
      );
      entry.body.position.set(...next);
    }

    world.step(FIXED_DT);
    for (const entry of dynamicEntries) {
      entry.positions.push({ frame, value: entry.body.position.toArray() });
    }
  }

  for (const entry of dynamicEntries) {
    entry.obj.positionKeyframes = entry.positions;
  }

  return { bakedObjectIds: dynamicEntries.map((e) => e.obj.id) };
}
