%module rt

%{
    #define SWIG_FILE_WITH_INIT
    #include "rt.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* cameras, int d1, int f1)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* lights, int d2, int f2)}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* obj_pos, int obj_num1, int d3)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* obj_amb, int obj_num2, int d4)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* obj_diff, int obj_num3, int d5)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* obj_spec, int obj_num4, int d6)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* pix_loc, int n1, int d7)}

%apply (float* IN_ARRAY1, int DIM1) {(float* obj_size, int obj_num5)}
%apply (float* IN_ARRAY1, int DIM1) {(float* obj_shine, int obj_num6)}
%apply (float* IN_ARRAY1, int DIM1) {(float* obj_refl, int obj_num7)}

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* pixels, int n, int m)}

%include "rt.h"

