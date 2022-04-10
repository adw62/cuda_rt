from rt import rt
import numpy as np
import matplotlib.pyplot as plt

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
pixels = np.zeros([1572864, 3], 'f')
x = 1536
y = 1024

pix_loc = np.array([[i, j] for i in np.linspace(-1, 1, x) for j in np.linspace(-1, 1, y)], 'f')
obj_pos = np.array([[-0.2, 0, -1],
                    [0.1, -0.3, 0],
                    [-0.3, 0, 0],
                    [0, -9000, 0]], 'f')
cameras = np.array([[0, 0, 1]], 'f')
lights = np.array([[5, 5, i] for i in np.linspace(0, 5, 200)], 'f')
for i, light in enumerate(lights):
    print(i)
    if i < 100:
        obj_pos[0][0] += 0.01
        obj_pos[0][1] += 0.001
        obj_pos[1][1] += 0.01
    else:
        obj_pos[0][0] -= 0.01
        obj_pos[1][1] -= 0.01
    rt(cameras, np.array([light]), obj_pos, obj_amb, obj_diff, obj_spec, obj_size, obj_shine, obj_refl, pixels, pix_loc)
    pixels = np.clip(pixels, 0, 1)
    pixels.shape = (x, y, 3)
    plt.imsave(f'./img/{i:03}_img.png', pixels)
    pixels = pixels*0.0
    pixels.shape = (1572864, 3)

