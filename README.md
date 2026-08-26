# cuda_rt

# Notes on requirements

pip install swig

sudo apt install build-essential

conda install -c conda-forge cudatoolkit-dev

sudo apt-get install python3-dev

pip3 install opencv-python

sudo apt-get install libgl1-mesa-glx

pip3 install flask

# To build

swig -python -c++ rt.i

nvcc --compiler-options '-fPIC' -c rt.cu rt_wrap.cxx -I/home/a/miniconda3/include/python3.9/ -I/home/a/miniconda3/lib/python3.9/site-packages/numpy/core/include

nvcc -shared rt.o rt_wrap.o -o _rt.so

# To run

python main.py [scenes/some_scene.json]

Defaults to `scenes/demo.json` if no scene is given. Writes rendered frames to `img/`, then
`video.avi` (full quality) and `video.gif` (downscaled, for sharing/embedding -- GitHub renders
`.gif` inline but not `.avi`).

![video](assets/demo.gif)

# Rendering features

- **Primitives**: spheres, boxes, and (infinite) planes, each with per-object ambient/diffuse/
  specular color, shininess, and mirror reflectivity; boxes and rolling spheres also carry a
  quaternion rotation.
- **Transparency**: any primitive can be made transparent (`transparency` 0-1, plus `ior` for the
  index of refraction). Glass is traced properly, not alpha-blended -- each transparent hit splits
  into a real reflected ray and a real Snell's-law-refracted ray, weighted by Fresnel reflectance,
  so you get actual lensing/distortion of whatever's behind the glass.
- **Colored glass**: an object's own `diffusion` color doubles as its glass tint -- near-white
  gives clear glass, a saturated color tints everything seen through it (and its cast shadow) that
  color, darkening further with each additional glass surface a ray passes through.
- **Tinted shadows through glass**: shadow rays walk through transparent occluders instead of
  stopping dead at the first hit, so glass casts a lighter, color-tinted shadow instead of a solid
  black one, same as it renders from the camera.
- **Caustics (photon mapping)**: a separate light-side pass emits photons from the light and
  traces them through any transparent object (reusing the same Snell's-law/Fresnel code as the
  camera rays), gathering them at shading time -- so glass actually focuses light into a bright
  caustic patch with a darker halo around it, instead of a uniformly dimmed shadow.
- **Checkered rolling spheres**: a sphere can be marked `checker` in the scene JSON (or via the
  editor's "Checkered" toggle). Its rotation is then driven kinematically -- rolling without
  slipping, purely from its own position track -- so you can see it actually spin as it moves,
  with a cheap procedural octant checker pattern to make the spin visible on the render (and a
  similar, non-matching checker texture in the editor's live preview).

# Scene editor

A browser-based editor for building scenes without hand-editing JSON: fly the camera around a
rasterized Three.js preview, spawn/move/rotate/scale sphere, box, and plane objects, keyframe
positions on a timeline, and bake a physics simulation (per-object gravity/collision, initial
velocity) into keyframes before running the real CUDA render.

The editor's third-party JS (three.js, cannon-es) isn't committed to the repo -- fetch it once with:

bash editor/fetch-vendor.sh

python server.py

Then open http://127.0.0.1:5000 in a browser.

- **Viewport**: orbit to look around; "Fly Camera" flies with WASD + mouse and moves the actual
  render camera; `K` inserts a camera keyframe at the current frame while flying.
- **Outliner / Inspector**: spawn spheres/boxes/planes; set material (including Transparency and
  IOR for glass), size, a sphere's `Checkered` flag, and per-object `Gravity` / `Collision` flags
  plus initial velocity (with optional randomization) for physics.
- **Gizmos**: `1` / `2` / `3` (or the toolbar) switch Move / Rotate / Scale.
- **Timeline**: scrub, drag, or delete position keyframes; play/pause.
- **Bake Physics**: runs the simulation and writes the result as position keyframes.
- **Render**: set the output resolution (Preview/Final presets or exact numbers), then submit the
  scene to the real CUDA renderer as a background job with progress and a preview.

Scenes are plain JSON files under `scenes/` -- `scene_io.py` loads them the same way whether
hand-written or exported from the editor. `scenes/demo.json` and `scenes/primitives_demo.json`
are examples.


