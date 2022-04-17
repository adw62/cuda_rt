from rt import rt
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

#Transparnacy
#complex camera moves

obj_amb = np.array([[0.1, 0, 0],
                    [0.1, 0, 0.1],
                    [0, 0.1, 0],
                    [0.1, 0.1, 0.1]], 'f')
obj_diff = np.array([[0.7, 0, 0],
                     [0.7, 0.0, 0.7],
                     [0, 0.6, 0],
                     [0.6, 0.6, 0.6]], 'f')
obj_spec = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]], 'f')
obj_size = np.array([0.7,
                     0.1,
                     0.15,
                     9000 - 0.7], 'f')
obj_shine = np.array([100,
                      100,
                      100,
                      100], 'f')
obj_refl = np.array([0.5,
                     0.5,
                     0.5,
                     0.5], 'f')
x = 1024
y = 1536
xy = 1572864
pixels = np.zeros([xy, 3], 'f')

pix_loc = np.array([[j, i] for i in np.linspace(1, -1, x) for j in np.linspace(-1, 1, y)], 'f')
obj_pos = np.array([[-0.2, 0, -1],
                    [0.1, -0.3, 0],
                    [-0.3, 0, 0],
                    [0, -9000, 0]], 'f')

frames = 500
cameras = np.array([[[0, 0, 3], [0, i, 0], [-0.2, 0, -1]] for i in np.linspace(0, 6, frames)], 'f')
lights = np.array([[i, 5, 5] for i in np.linspace(-10, 10, frames)], 'f')
for i, (camera, light) in enumerate(zip(cameras, lights)):
    print(i)
    if i < int(frames*0.3):
        obj_pos[0][1] += 0.001
        obj_pos[1][1] += 0.01
    else:
        obj_pos[1][1] -= 0.01
    rt(camera, np.array([light]), obj_pos, obj_amb, obj_diff, obj_spec, obj_size, obj_shine, obj_refl, pixels, pix_loc)
    pixels = np.clip(pixels, 0, 1)
    pixels.shape = (x, y, 3)
    plt.imsave(f'./img/{i:03}_img.png', pixels)
    pixels = pixels*0.0
    pixels.shape = (xy, 3)

