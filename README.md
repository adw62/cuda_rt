# cuda_rt

# Notes on requirements

pip install swig

sudo apt install build-essential

conda install -c conda-forge cudatoolkit-dev

sudo apt-get install python3-dev

pip3 install opencv-python

sudo apt-get install libgl1-mesa-glx

# To build

swig -python -c++ rt.i

nvcc --compiler-options '-fPIC' -c rt.cu rt_wrap.cxx -I/home/a/miniconda3/include/python3.9/ -I/home/a/miniconda3/lib/python3.9/site-packages/numpy/core/include

nvcc -shared rt.o rt_wrap.o -o _rt.so

# To run

python main.py

![video](https://user-images.githubusercontent.com/38112687/167273814-3c1bccad-3996-497a-a499-e88ecb39c672.gif)


