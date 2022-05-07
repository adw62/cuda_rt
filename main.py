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
        self.obj_amb = None
        self.obj_diff = None
        self.obj_spec = None
        self.obj_size = None
        self.obj_shine = None
        self.obj_refl = None
        self.obj_pos = None

    def add_object(self, obj):
        assert isinstance(obj, Sphere)
        self.objects.append(obj)

    def build(self):
        self.obj_amb = np.array([x.ambient for x in self.objects], 'f')
        self.obj_diff = np.array([x.diffusion for x in self.objects], 'f')
        self.obj_spec = np.array([x.specular for x in self.objects], 'f')
        self.obj_size = np.array([x.size for x in self.objects], 'f')
        self.obj_shine = np.array([x.shine for x in self.objects], 'f')
        self.obj_refl = np.array([x.reflection for x in self.objects], 'f')
        self.obj_pos = np.array([x.positions for x in self.objects], 'f')
        self.obj_pos = np.transpose(self.obj_pos, (1, 0, 2))
        print('Built {} object(s)'.format(len(self.objects)))

    def render(self):
        for i in range(self.frames):
            print('Rendering frame {}/{}'.format(i, self.frames))
            rt(self.cameras[i], self.lights[i], self.obj_pos[i], self.obj_amb, self.obj_diff, self.obj_spec,
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

class Camera():
    def __init__(self, start, stop, frames):
        x = straight_line(start[0], stop[0], frames)
        y = straight_line(start[1], stop[1], frames)
        z = straight_line(start[2], stop[2], frames)
        self.traj = np.array([[i, j, k] for i, j, k in zip(x, y, z)], 'f')

class Light():
    def __init__(self, start, stop, frames):
        line = straight_line(start, stop, frames)
        self.traj = np.array([[i] for i in line], 'f')

class Sphere():
    def __init__(self, size, shine, reflection, ambient, diffusion, specular, positions):
        self.ambient = ambient
        self.diffusion = diffusion
        self.specular = specular
        self.size = size
        self.shine = shine
        self.reflection = reflection
        self.positions = positions

def straight_line(start, stop, frames):
    x1 = np.linspace(start[0], stop[0], frames)
    y1 = np.linspace(start[1], stop[1], frames)
    z1 = np.linspace(start[2], stop[2], frames)
    return [[x, y, z] for x, y, z in zip(x1, y1, z1)]


def multi_line(points, frame_chuncks, frames):
    assert sum(frame_chuncks) == frames
    assert len(frame_chuncks) == (len(points) - 1)
    line = []
    for i, (point, f) in enumerate(zip(points, frame_chuncks)):
        if i < len(points):
            line.extend(straight_line(point, points[i + 1], f))
    return line


if __name__ == '__main__':
    frames = 100
    a = Sphere(0.7,
               100,
               0.5,
               [0.1, 0, 0],
               [0.7, 0, 0],
               [1, 1, 1],
               straight_line([-0.2, 0, -1], [-0.2, 0, -1], frames))
    b = Sphere(0.1,
               100,
               0.5,
               [0.1, 0, 0.1],
               [0.7, 0.0, 0.7],
               [1, 1, 1],
               multi_line([[0.1, -0.3, 0], [1, -0.3, 0], [1, 1, 0], [0.1, 1, 0], [0.1, -0.3, 0]],
                          [int(frames/4), int(frames/4), int(frames/4), int(frames/4)], frames))
    c = Sphere(0.15,
               100,
               0.5,
               [0, 0.1, 0],
               [0, 0.6, 0],
               [1, 1, 1],
               straight_line([-0.3, 0, 0], [-0.3, 0, 0], frames))
    d = Sphere(9000 - 0.7,
               100,
               0.5,
               [0.1, 0.1, 0.1],
               [0.6, 0.6, 0.6],
               [1, 1, 1],
               straight_line([0, -9000, 0], [0, -9000, 0], frames))

    cameras = Camera([[0, 0, 2], [0, 0, 0], [-0.2, 0, -1]], [[0, 0, 3], [0, 0, 0], [-0.2, 0, -1]], frames).traj
    lights = Light([-10, 5, 5], [10, 5, 5], frames).traj

    S = Scene(1024, 1536, cameras, lights, frames)
    S.add_object(a)
    S.add_object(b)
    S.add_object(c)
    S.add_object(d)

    S.build()
    S.render()
    S.make_video()
