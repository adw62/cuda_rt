from rt import rt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

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
        self.obj_pos = np.array([x.positions for x in self.objects], 'f')
        self.obj_pos = np.transpose(self.obj_pos, (1, 0, 2))
        self.obj_param1 = np.array([x.param1 or [0, 0, 0] for x in self.objects], 'f')
        self.obj_rot = np.array([x.rotation or [0, 0, 0, 1] for x in self.objects], 'f')
        print('Built {} object(s)'.format(len(self.objects)))

    def render(self):
        for i in range(self.frames):
            print('Rendering frame {}/{}'.format(i, self.frames))
            rt(self.cameras[i], self.lights[i], self.obj_type, self.obj_pos[i], self.obj_param1, self.obj_rot,
               self.obj_amb, self.obj_diff, self.obj_spec,
               self.obj_size, self.obj_shine, self.obj_refl, self.pixels, self.pix_loc)
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

TYPE_IDS = {'sphere': 0, 'plane': 1, 'box': 2}


class Primitive():
    def __init__(self, type, size, shine, reflection, ambient, diffusion, specular, positions, param1=None, rotation=None):
        assert type in TYPE_IDS
        self.type = type
        self.ambient = ambient
        self.diffusion = diffusion
        self.specular = specular
        self.size = size
        self.shine = shine
        self.reflection = reflection
        self.positions = positions
        # meaning depends on type: unused for sphere, normal for plane, half-extents for box
        self.param1 = param1
        # quaternion (x,y,z,w); box-only, None means identity (axis-aligned)
        self.rotation = rotation


if __name__ == '__main__':
    import sys
    import scene_io

    scene_path = sys.argv[1] if len(sys.argv) > 1 else 'scenes/demo.json'
    S = scene_io.load_scene(scene_path)

    S.build()
    S.render()
    S.make_video()
