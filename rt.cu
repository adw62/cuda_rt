#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cassert>
#include <math.h>

using namespace std;

inline __device__ float3 operator-(float3& a, float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ void operator+=(float3& a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __device__ void operator*=(float3& a, float3 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __device__ float3 operator+(float3& a, float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator*(float3& a, float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ float3 operator*(float& a, float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __device__ float dot(float3& a, float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __device__ float3 reflect(float3& vec, float3& axis) {
    float tmp = 2*dot(vec, axis);
    float3 tmp1 = tmp*axis;
    return vec - tmp1;
}

inline __device__ float norm_(float3& a) {
    float mag = sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
    return mag;
}

inline __device__ float3 normalize(float3& a) {
    float mag = norm_(a);
    return make_float3(a.x/mag, a.y/mag, a.z/mag);
}

inline __device__ float3 rotate(float3& a, float3& cam_angle, float3& rot_center) {
    float3 new_pix = {0, 0, 0};
    //rotate around z axis
    float adjusted_x = a.x - rot_center.x;
    float adjusted_y = a.y - rot_center.y;
    float adjusted_z = a.z - rot_center.z;
    new_pix.x = a.x*cos(cam_angle.z) + a.y*sin(cam_angle.z);
    new_pix.y = -1*a.x*sin(cam_angle.z) + a.y*cos(cam_angle.z);
    new_pix.z = a.z;

    //rotate around y axis
    new_pix.x = new_pix.x*cos(cam_angle.y) - new_pix.z*sin(cam_angle.y);
    new_pix.y = new_pix.y;
    new_pix.z = new_pix.x*sin(cam_angle.y) + new_pix.z*cos(cam_angle.y);

    //rotate around x axis
    new_pix.x = new_pix.x;
    new_pix.y = new_pix.y*cos(cam_angle.x) + new_pix.z*sin(cam_angle.x);
    new_pix.z = -1*new_pix.y*sin(cam_angle.x) + new_pix.z*cos(cam_angle.x);

    return new_pix;
}

inline __device__ float intersect(float3& center, float& size, float3& origin, float3& direction) {
    float3 tmp = origin-center;
    float b = 2*dot(direction, tmp);
    tmp = origin-center;
    float c = pow(norm_(tmp), 2) - pow(size, 2);
    float delta = pow(b, 2) - 4*c;
    if (delta > 0) {
        float t1 = (-b+sqrt(delta))/2;
        float t2 = (-b-sqrt(delta))/2;
        if (t1 > 0 and t2 > 0){
            if (t1 < t2){
                return t1;
            }
            return t2;
        }
    }
    return 100000000;
}

inline __device__ void print(float3& a) {
    printf( "{%6.4lf, ", a.x);
    printf( "%6.4lf, ", a.y);
    printf( "%6.4lf}\n", a.z);
}

inline __device__ float2 nearest_intersected_object(float3 *obj_pos, float *obj_size,
 float3& origin, float3 direction, int& num_obj) {
    float nearest_obj = num_obj+1;
    float min_dist = 100000000;
    for (int j = 0; j < num_obj; ++j) {
        float3 center = obj_pos[j];
        float size = obj_size[j];
        float calc_dist = intersect(center, size, origin, direction);
        if (calc_dist < min_dist){
            nearest_obj = j;
            min_dist = calc_dist;
        }
    }
    return float2 {nearest_obj, min_dist};
}

__global__
void rt_kernel(float3 *dpixels, float *dobj_size, float *dobj_shine, float *dobj_refl,
 float3 *dobj_pos, float3 *dobj_amb, float3 *dobj_diff, float3 *dobj_spec,
  float3 *dcameras, float3 *dlights, float2 *dpix_loc, int num_obj) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    float3 amb_l = {1, 1, 1};
    float3 diff_l = {1, 1, 1};
    float3 spec_l = {1, 1, 1};

    float3 light = dlights[0];
    float tmp1 = 0;
    float3 tmp = {0, 0, 0};

    float3 cam_angle = dcameras[1];
    float3 cam = dcameras[0];

    cam = rotate(cam, cam_angle, dcameras[2]);
    float3 origin = cam;

    //the screen is tied to the camera
    tmp = {dpix_loc[i].x+cam.x, dpix_loc[i].y+cam.y, 0+(cam.z-1)};
    float3 pixel = rotate(tmp, cam_angle, dcameras[2]);
    float3 direction = pixel-origin;
    direction = normalize(direction);

    int max_depth = 3;
    float reflection = 1;
    float intersection_to_light_distance = 0;
    float shift = 0.00001;
    float2 res = {0, 0};
    float2 res1 = {0, 0};
    float3 intersection = {0, 0, 0};
    float3 normal_to_surface = {0, 0, 0};
    float3 shifted_point = {0, 0, 0};
    float3 intersection_to_light = {0, 0, 0};
    float3 illumination = {0, 0, 0};
    float3 intersection_to_camera = {0, 0, 0};
    float3 H = {0, 0, 0};
    float3 color = {0, 0, 0};

    //loop over a depth of ray casts
    for (int k = 0; k < max_depth; ++k) {
        //look for distance to objects
        //{nearest_obj, min_dist}
        res = nearest_intersected_object(dobj_pos, dobj_size, origin, direction, num_obj);
        int nearest_obj = res.x;
        if (nearest_obj > num_obj){
            break;
        }
        tmp = res.y*direction;
        intersection = origin + tmp;
        tmp = intersection - dobj_pos[nearest_obj];
        normal_to_surface  = normalize(tmp);
        tmp = shift*normal_to_surface;
        shifted_point = intersection + tmp;
        tmp = light - shifted_point;
        intersection_to_light = normalize(tmp);

        res1 = nearest_intersected_object(dobj_pos, dobj_size, shifted_point, intersection_to_light, num_obj);
        tmp = light-intersection;
        intersection_to_light_distance = norm_(tmp);
        if (res1.y < intersection_to_light_distance) {
            break;
        }
        illumination = {0, 0, 0};
        //amb
        tmp = amb_l*dobj_amb[nearest_obj];
        illumination += tmp;

        //diffuse
        tmp1 = dot(intersection_to_light, normal_to_surface);
        tmp = tmp1*diff_l;
        tmp = dobj_diff[nearest_obj]*tmp;
        illumination += tmp;

        //specular
        tmp = cam - intersection;
        intersection_to_camera = normalize(tmp);
        tmp = intersection_to_light + intersection_to_camera;
        H = normalize(tmp);
        tmp = spec_l * dobj_spec[nearest_obj];
        tmp1 = pow(dot(normal_to_surface, H), (dobj_shine[nearest_obj]/4));
        illumination += tmp1 * tmp;

        //reflection
        color += reflection*illumination;
        reflection *= dobj_refl[nearest_obj];

        origin = shifted_point;
        direction = reflect(direction, normal_to_surface);

    //pixels is array of float need to be array of float3 to store rgb
    dpixels[i] = color;
    }

}

void rt(float *cameras, int f1, int d1, float *lights, int f2, int d2,
                float *obj_pos, int obj_num1, int d3,
                float *obj_amb, int obj_num2, int d4,
                float *obj_diff, int obj_num3, int d5,
                float *obj_spec, int obj_num4, int d6,
                float *obj_size, int obj_num5,
                float *obj_shine, int obj_num6,
                float *obj_refl, int obj_num7,
                float *pixels, int n, int m,
                float *pix_loc, int n1, int d7) {

    //check number of frames for camera and light are the same
    //assert(f1==f2);
    //check number of objects the same in all data input
    assert(obj_num1==obj_num2);
    assert(obj_num2==obj_num3);
    assert(obj_num3==obj_num4);
    assert(obj_num4==obj_num5);
    assert(obj_num5==obj_num5);
    assert(obj_num6==obj_num7);
    //check all data has 3 dims
    assert(d1==3);
    assert(d1==d2);
    assert(d2==d3);
    assert(d3==d4);
    assert(d4==d5);
    assert(d5==d6);
    //check pixel dim is same between color array and location array
    assert(n==n1);

    float3 *dpixels, *dobj_pos, *dobj_amb, *dobj_diff, *dobj_spec, *dcameras, *dlights;
    float *dobj_size, *dobj_shine, *dobj_refl;
    float2 *dpix_loc;

    cudaMalloc(&dobj_size, obj_num1*sizeof(float));
    cudaMalloc(&dobj_shine, obj_num1*sizeof(float));
    cudaMalloc(&dobj_refl, obj_num1*sizeof(float));

    cudaMalloc(&dpixels, n*sizeof(float3));
    cudaMalloc(&dpix_loc, n*sizeof(float2));
    cudaMalloc(&dobj_pos, obj_num1*sizeof(float3));
    cudaMalloc(&dobj_amb, obj_num1*sizeof(float3));
    cudaMalloc(&dobj_diff, obj_num1*sizeof(float3));
    cudaMalloc(&dobj_spec, obj_num1*sizeof(float3));
    cudaMalloc(&dcameras, f1*sizeof(float3));
    cudaMalloc(&dlights, f1*sizeof(float3));

    //copy device arrays to device
    cudaMemcpy(dobj_size, obj_size, obj_num1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_shine, obj_shine, obj_num1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_refl, obj_refl, obj_num1*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dpixels, pixels, n*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dpix_loc, pix_loc, n*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_pos, obj_pos, obj_num1*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_amb, obj_amb, obj_num1*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_diff, obj_diff, obj_num1*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_spec, obj_spec, obj_num1*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dcameras, cameras, f1*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dlights, lights, f1*sizeof(float3), cudaMemcpyHostToDevice);

    rt_kernel<<<n/512, 512>>>(dpixels, dobj_size, dobj_shine, dobj_refl,
     dobj_pos, dobj_amb, dobj_diff, dobj_spec, dcameras, dlights, dpix_loc, obj_num1);

    cudaMemcpy(pixels, dpixels, n*sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(dpixels);
    cudaFree(dpix_loc);
    cudaFree(dobj_pos);
    cudaFree(dobj_amb);
    cudaFree(dobj_diff);
    cudaFree(dobj_spec);
    cudaFree(dcameras);
    cudaFree(dlights);
    cudaFree(dobj_size);
    cudaFree(dobj_shine);
    cudaFree(dobj_refl);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
            cudaGetErrorString(cudaerr));
}
