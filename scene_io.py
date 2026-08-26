import json
import math
import numpy as np
from main import Scene, Primitive


def _lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]


def _normalize(v):
    mag = sum(c * c for c in v) ** 0.5
    return [c / mag for c in v]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _quat_mul(a, b):
    # a * b, both (x, y, z, w)
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _axis_angle_quat(axis, angle):
    half = angle / 2.0
    s = math.sin(half)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)]


def rolling_quaternions(positions, radius):
    """Rolling-without-slipping quaternion track from a per-frame position
    track: each step rotates about (up x delta) by arclength/radius. Must
    match computeRollingQuaternions in scene-model.js frame-for-frame so the
    editor preview spin agrees with what the CUDA render actually produces."""
    up = [0, 1, 0]
    quat = [0, 0, 0, 1]
    out = [quat]
    for prev, cur in zip(positions, positions[1:]):
        delta = [cur[i] - prev[i] for i in range(3)]
        dist = sum(c * c for c in delta) ** 0.5
        if dist > 1e-9:
            axis = _cross(up, delta)
            axis_mag = sum(c * c for c in axis) ** 0.5
            if axis_mag > 1e-9:
                axis = [c / axis_mag for c in axis]
                dq = _axis_angle_quat(axis, dist / radius)
                quat = _quat_mul(dq, quat)
                mag = sum(c * c for c in quat) ** 0.5
                quat = [c / mag for c in quat]
        out.append(quat)
    return out


def sample_track(keyframes, frame, normalize=False):
    """Piecewise-linear lookup over a sorted set of {frame, value} keyframes.
    Clamps to the first/last keyframe outside their range. `normalize` re-unit-lengths
    the interpolated result, for direction tracks (camera forward/up)."""
    kfs = sorted(keyframes, key=lambda k: k['frame'])
    if len(kfs) == 1 or frame <= kfs[0]['frame']:
        value = kfs[0]['value']
    elif frame >= kfs[-1]['frame']:
        value = kfs[-1]['value']
    else:
        for a, b in zip(kfs, kfs[1:]):
            if a['frame'] <= frame <= b['frame']:
                span = b['frame'] - a['frame']
                t = 0.0 if span == 0 else (frame - a['frame']) / span
                value = _lerp(a['value'], b['value'], t)
                break
    return _normalize(value) if normalize else value


def _sample_all_frames(keyframes, frames, normalize=False):
    return [sample_track(keyframes, f, normalize=normalize) for f in range(frames)]


def load_scene(source):
    """Build a Scene from a scene JSON file path, or an already-parsed dict."""
    data = source if isinstance(source, dict) else json.load(open(source))
    frames = data['frames']
    res = data['resolution']

    objects = []
    for obj in data['objects']:
        pos_kfs = [{'frame': k['frame'], 'value': k['pos']} for k in obj['position_keyframes']]
        positions = _sample_all_frames(pos_kfs, frames)
        obj_type = obj.get('type', 'sphere')
        checkered = obj_type == 'sphere' and obj.get('checker')
        if obj_type == 'box':
            param1 = obj['half_extents']
        elif obj_type == 'plane':
            param1 = obj['normal']
        elif checkered:
            param1 = [1.0, 0.0, 0.0]  # sphere-only: checker on/off flag
        else:
            param1 = None

        if checkered:
            rotations = rolling_quaternions(positions, obj.get('size', 1.0))
        else:
            static = obj.get('rotation') or [0, 0, 0, 1]
            rotations = [static] * frames

        objects.append(Primitive(obj_type, obj.get('size', 1.0), obj['shine'], obj['reflection'],
                                  obj['ambient'], obj['diffusion'], obj['specular'],
                                  positions, param1=param1, rotation=rotations,
                                  transparency=obj.get('transparency', 0.0), ior=obj.get('ior', 1.5)))

    cam_kfs = data['camera']['keyframes']
    cam_pos = _sample_all_frames([{'frame': k['frame'], 'value': k['position']} for k in cam_kfs], frames)
    cam_fwd = _sample_all_frames([{'frame': k['frame'], 'value': k['forward']} for k in cam_kfs], frames, normalize=True)
    cam_up = _sample_all_frames([{'frame': k['frame'], 'value': k['up']} for k in cam_kfs], frames, normalize=True)
    cameras = np.array([[p, f, u] for p, f, u in zip(cam_pos, cam_fwd, cam_up)], 'f')

    light_kfs = [{'frame': k['frame'], 'value': k['position']} for k in data['light']['keyframes']]
    light_pos = _sample_all_frames(light_kfs, frames)
    lights = np.array([[p] for p in light_pos], 'f')

    scene = Scene(res['x'], res['y'], cameras, lights, frames)
    for obj in objects:
        scene.add_object(obj)
    return scene
