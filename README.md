# cuda_rt

swig -python -c++ rt.i
nvcc --compiler-options '-fPIC' -c rt.cu rt_wrap.cxx -I/home/a/miniconda3/include/python3.9/ -I/home/a/miniconda3/lib/python3.9/site-packages/numpy/core/include
nvcc -shared rt.o rt_wrap.o -o _rt.so
