from rt import rt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

class Scene():
    def __init__(self, x, y, cameras, lights, frames):
        self.x = x
        self.y = y
        self.pix_loc = np.array([[j, i] for i in np.linspace(1, -1, x) for j in np.linspace(-1, 1, y)], 'f')
        self.pixels = np.zeros([self.x * self.y, 3], 'f')
        self.cameras = cameras
        self.lights = lights
        self.frames = frames
        self.objects = []
        self.obj_type = None
        self.obj_amb = None
        self.obj_diff = None
        self.obj_spec = None
        self.obj_size = None
        self.obj_shine = None
        self.obj_refl = None
        self.obj_transparency = None
        self.obj_ior = None
        self.obj_pos = None
        self.obj_param1 = None
        self.obj_rot = None

    def add_object(self, obj):
        assert isinstance(obj, Primitive)
        self.objects.append(obj)

    def build(self):
        self.obj_type = np.array([TYPE_IDS[x.type] for x in self.objects], 'i')
        self.obj_amb = np.array([x.ambient for x in self.objects], 'f')
        self.obj_diff = np.array([x.diffusion for x in self.objects], 'f')
        self.obj_spec = np.array([x.specular for x in self.objects], 'f')
        self.obj_size = np.array([x.size for x in self.objects], 'f')
        self.obj_shine = np.array([x.shine for x in self.objects], 'f')
        self.obj_refl = np.array([x.reflection for x in self.objects], 'f')
        self.obj_transparency = np.array([x.transparency for x in self.objects], 'f')
        self.obj_ior = np.array([x.ior for x in self.objects], 'f')
        self.obj_pos = np.array([x.positions for x in self.objects], 'f')
        self.obj_pos = np.transpose(self.obj_pos, (1, 0, 2))
        self.obj_param1 = np.array([x.param1 or [0, 0, 0] for x in self.objects], 'f')
        # per-frame like obj_pos: static objects just repeat the same quaternion
        # every frame, a rolling checkered sphere gets one that actually spins
        self.obj_rot = np.array([x.rotations for x in self.objects], 'f')
        self.obj_rot = np.transpose(self.obj_rot, (1, 0, 2))
        print('Built {} object(s)'.format(len(self.objects)))

    def render(self):
        for i in range(self.frames):
            print('Rendering frame {}/{}'.format(i, self.frames))
            rt(self.cameras[i], self.lights[i], self.obj_type, self.obj_pos[i], self.obj_param1, self.obj_rot[i],
               self.obj_amb, self.obj_diff, self.obj_spec,
               self.obj_size, self.obj_shine, self.obj_refl, self.obj_transparency, self.obj_ior,
               self.pixels, self.pix_loc)
            self.pixels = np.clip(self.pixels, 0, 1)
            self.pixels.shape = (self.x, self.y, 3)
            plt.imsave(f'./img/{i:03}_img.png', self.pixels)
            self.pixels = self.pixels * 0.0
            self.pixels.shape = (self.x*self.y, 3)

    def make_video(self):
        print('Assembling frames into avi')
        image_folder = 'img'
        video_name = 'video.avi'
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 18, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.destroyAllWindows()
        video.release()

    def make_gif(self, path='video.gif', fps=18, max_width=480):
        # Lightweight companion to the .avi -- downscaled so it's small enough
        # to embed inline in a README/PR (GitHub renders .gif inline, not .avi).
        print('Assembling frames into gif')
        image_folder = 'img'
        images = sorted(img for img in os.listdir(image_folder) if img.endswith('.png'))
        frames = []
        for name in images:
            im = Image.open(os.path.join(image_folder, name)).convert('RGB')
            if max_width and im.width > max_width:
                new_height = int(im.height * (max_width / im.width))
                im = im.resize((max_width, new_height), Image.LANCZOS)
            frames.append(im)
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 / fps, loop=0)

TYPE_IDS = {'sphere': 0, 'plane': 1, 'box': 2}


class Primitive():
    def __init__(self, type, size, shine, reflection, ambient, diffusion, specular, positions, param1=None, rotation=None,
                 transparency=0.0, ior=1.5):
        assert type in TYPE_IDS
        self.type = type
        self.ambient = ambient
        self.diffusion = diffusion
        self.specular = specular
        self.size = size
        self.shine = shine
        self.reflection = reflection
        self.positions = positions
        # 0 = opaque (default, existing behavior unchanged), 1 = fully clear;
        # splits into a Fresnel-weighted reflect/refract ray pair in rt.cu
        self.transparency = transparency
        # index of refraction, only meaningful once transparency > 0
        self.ior = ior
        # meaning depends on type: checker on/off for sphere, normal for plane, half-extents for box
        self.param1 = param1
        # one quaternion (x,y,z,w) per frame; box is a static pose repeated every
        # frame, a rolling checkered sphere gets a spinning track, None means identity
        self.rotations = rotation or [[0, 0, 0, 1]] * len(positions)


if __name__ == '__main__':
    import sys
    import scene_io

    scene_path = sys.argv[1] if len(sys.argv) > 1 else 'scenes/demo.json'
    S = scene_io.load_scene(scene_path)

    S.build()
    S.render()
    S.make_video()
    S.make_gif()
