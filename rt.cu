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

inline __device__ float3 cross(float3& a, float3& b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
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

#define TYPE_SPHERE 0
#define TYPE_PLANE 1
#define TYPE_BOX 2
#define MISS 100000000.f

inline __device__ float intersect_sphere(float3& center, float& size, float3& origin, float3& direction) {
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
    return MISS;
}

// point is any point on the (infinite) plane, normal need not be unit length
inline __device__ float intersect_plane(float3& point, float3& normal, float3& origin, float3& direction) {
    float denom = dot(normal, direction);
    if (fabsf(denom) < 1e-6f) {
        return MISS;
    }
    float3 diff = point-origin;
    float t = dot(diff, normal)/denom;
    return t > 1e-4f ? t : MISS;
}

inline __device__ float4 quat_conj(float4& q) {
    return make_float4(-q.x, -q.y, -q.z, q.w);
}

// standard two-cross-product quaternion rotation (no trig, no near-zero-angle
// special case -- rotating by the identity quaternion (0,0,0,1) is an exact
// floating-point no-op since qv is (0,0,0) and every cross-product term is 0)
inline __device__ float3 quat_rotate(float3& v, float4& q) {
    float3 qv = make_float3(q.x, q.y, q.z);
    float3 t_raw = cross(qv, v);
    float two = 2.f;
    float3 t = two*t_raw;
    float3 qwt = q.w*t;
    float3 qvt = cross(qv, t);
    float3 sum1 = v+qwt;
    return sum1+qvt;
}

// box, optionally rotated about its center (identity quaternion = axis-aligned)
inline __device__ float intersect_box(float3& center, float3& half_extents, float4& rotation, float3& origin, float3& direction) {
    float3 rel = origin-center;
    float4 inv_rot = quat_conj(rotation);
    float3 local_origin = quat_rotate(rel, inv_rot);
    float3 local_direction = quat_rotate(direction, inv_rot);
    float o[3] = {local_origin.x, local_origin.y, local_origin.z};
    float d[3] = {local_direction.x, local_direction.y, local_direction.z};
    float h[3] = {half_extents.x, half_extents.y, half_extents.z};

    float tmin = -MISS;
    float tmax = MISS;
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(d[axis]) < 1e-8f) {
            if (o[axis] < -h[axis] || o[axis] > h[axis]) {
                return MISS;
            }
            continue;
        }
        float t1 = (-h[axis]-o[axis])/d[axis];
        float t2 = (h[axis]-o[axis])/d[axis];
        if (t1 > t2) {
            float swap = t1; t1 = t2; t2 = swap;
        }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmin > tmax) {
            return MISS;
        }
    }
    if (tmin > 1e-4f) {
        return tmin;
    }
    if (tmax > 1e-4f) {
        return tmax;
    }
    return MISS;
}

inline __device__ float3 flip(float3& v) {
    return make_float3(-v.x, -v.y, -v.z);
}

// normal at a hit point, dispatched on object type. The sphere branch is
// exactly the original (pre-multi-primitive) normal computation, unchanged,
// to keep sphere-only scenes rendering identically.
inline __device__ float3 surface_normal(int type, float3& center, float3& param1, float4& rotation, float3& hit, float3& incoming) {
    if (type == TYPE_PLANE) {
        float3 n = normalize(param1);
        if (dot(n, incoming) > 0) {
            n = flip(n);
        }
        return n;
    }
    if (type == TYPE_BOX) {
        float3 rel = hit-center;
        float4 inv_rot = quat_conj(rotation);
        float3 local = quat_rotate(rel, inv_rot);
        float3 n = make_float3(local.x/param1.x, local.y/param1.y, local.z/param1.z);
        float ax = fabsf(n.x), ay = fabsf(n.y), az = fabsf(n.z);
        float3 local_normal;
        if (ax > ay && ax > az) {
            local_normal = make_float3(n.x > 0 ? 1.f : -1.f, 0.f, 0.f);
        } else if (ay > az) {
            local_normal = make_float3(0.f, n.y > 0 ? 1.f : -1.f, 0.f);
        } else {
            local_normal = make_float3(0.f, 0.f, n.z > 0 ? 1.f : -1.f);
        }
        return quat_rotate(local_normal, rotation);
    }
    float3 d = hit-center;
    return normalize(d);
}

inline __device__ void print(float3& a) {
    printf( "{%6.4lf, ", a.x);
    printf( "%6.4lf, ", a.y);
    printf( "%6.4lf}\n", a.z);
}

inline __device__ float2 nearest_intersected_object(int *obj_type, float3 *obj_pos, float3 *obj_param1, float4 *obj_rot, float *obj_size,
 float3& origin, float3 direction, int& num_obj) {
    float nearest_obj = num_obj+1;
    float min_dist = MISS;
    for (int j = 0; j < num_obj; ++j) {
        float3 center = obj_pos[j];
        float calc_dist;
        if (obj_type[j] == TYPE_PLANE) {
            float3 normal = obj_param1[j];
            calc_dist = intersect_plane(center, normal, origin, direction);
        } else if (obj_type[j] == TYPE_BOX) {
            float3 half_extents = obj_param1[j];
            float4 rotation = obj_rot[j];
            calc_dist = intersect_box(center, half_extents, rotation, origin, direction);
        } else {
            float size = obj_size[j];
            calc_dist = intersect_sphere(center, size, origin, direction);
        }
        if (calc_dist < min_dist){
            nearest_obj = j;
            min_dist = calc_dist;
        }
    }
    return float2 {nearest_obj, min_dist};
}

__global__
void rt_kernel(float3 *dpixels, float *dobj_size, float *dobj_shine, float *dobj_refl,
 int *dobj_type, float3 *dobj_pos, float3 *dobj_param1, float4 *dobj_rot, float3 *dobj_amb, float3 *dobj_diff, float3 *dobj_spec,
  float3 *dcameras, float3 *dlights, float2 *dpix_loc, int num_obj) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    float3 amb_l = {1, 1, 1};
    float3 diff_l = {1, 1, 1};
    float3 spec_l = {1, 1, 1};

    float3 light = dlights[0];
    float tmp1 = 0;
    float3 tmp = {0, 0, 0};

    //camera is a (position, forward, up) basis: right/up are re-derived each
    //frame so interpolated forward/up keyframes stay orthonormal
    float3 eye = dcameras[0];
    float3 forward = normalize(dcameras[1]);
    float3 up_raw = dcameras[2];
    float3 right_raw = cross(forward, up_raw);
    float3 right = normalize(right_raw);
    float3 up = cross(right, forward);

    float3 origin = eye;

    //the screen is tied to the camera basis
    float3 screen_x = dpix_loc[i].x*right;
    float3 screen_y = dpix_loc[i].y*up;
    tmp = forward+screen_x;
    float3 direction = tmp+screen_y;
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
        res = nearest_intersected_object(dobj_type, dobj_pos, dobj_param1, dobj_rot, dobj_size, origin, direction, num_obj);
        int nearest_obj = res.x;
        if (nearest_obj > num_obj){
            break;
        }
        tmp = res.y*direction;
        intersection = origin + tmp;
        normal_to_surface = surface_normal(dobj_type[nearest_obj], dobj_pos[nearest_obj], dobj_param1[nearest_obj], dobj_rot[nearest_obj], intersection, direction);
        tmp = shift*normal_to_surface;
        shifted_point = intersection + tmp;
        tmp = light - shifted_point;
        intersection_to_light = normalize(tmp);

        res1 = nearest_intersected_object(dobj_type, dobj_pos, dobj_param1, dobj_rot, dobj_size, shifted_point, intersection_to_light, num_obj);
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
        tmp = eye - intersection;
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
                int *obj_type, int obj_num0,
                float *obj_pos, int obj_num1, int d3,
                float *obj_param1, int obj_num1b, int d3b,
                float *obj_rot, int obj_num1c, int d3c,
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
    assert(obj_num0==obj_num1);
    assert(obj_num1==obj_num1b);
    assert(obj_num1==obj_num1c);
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
    assert(d3==d3b);
    assert(d3c==4);
    assert(d3==d4);
    assert(d4==d5);
    assert(d5==d6);
    //check pixel dim is same between color array and location array
    assert(n==n1);

    float3 *dpixels, *dobj_pos, *dobj_param1, *dobj_amb, *dobj_diff, *dobj_spec, *dcameras, *dlights;
    float4 *dobj_rot;
    float *dobj_size, *dobj_shine, *dobj_refl;
    int *dobj_type;
    float2 *dpix_loc;

    cudaMalloc(&dobj_type, obj_num1*sizeof(int));
    cudaMalloc(&dobj_size, obj_num1*sizeof(float));
    cudaMalloc(&dobj_shine, obj_num1*sizeof(float));
    cudaMalloc(&dobj_refl, obj_num1*sizeof(float));

    cudaMalloc(&dpixels, n*sizeof(float3));
    cudaMalloc(&dpix_loc, n*sizeof(float2));
    cudaMalloc(&dobj_pos, obj_num1*sizeof(float3));
    cudaMalloc(&dobj_param1, obj_num1*sizeof(float3));
    cudaMalloc(&dobj_rot, obj_num1*sizeof(float4));
    cudaMalloc(&dobj_amb, obj_num1*sizeof(float3));
    cudaMalloc(&dobj_diff, obj_num1*sizeof(float3));
    cudaMalloc(&dobj_spec, obj_num1*sizeof(float3));
    cudaMalloc(&dcameras, f1*sizeof(float3));
    cudaMalloc(&dlights, f1*sizeof(float3));

    //copy device arrays to device
    cudaMemcpy(dobj_type, obj_type, obj_num1*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_size, obj_size, obj_num1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_shine, obj_shine, obj_num1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_refl, obj_refl, obj_num1*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dpixels, pixels, n*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dpix_loc, pix_loc, n*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_pos, obj_pos, obj_num1*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_param1, obj_param1, obj_num1*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_rot, obj_rot, obj_num1*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_amb, obj_amb, obj_num1*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_diff, obj_diff, obj_num1*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_spec, obj_spec, obj_num1*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dcameras, cameras, f1*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dlights, lights, f1*sizeof(float3), cudaMemcpyHostToDevice);

    rt_kernel<<<n/512, 512>>>(dpixels, dobj_size, dobj_shine, dobj_refl,
     dobj_type, dobj_pos, dobj_param1, dobj_rot, dobj_amb, dobj_diff, dobj_spec, dcameras, dlights, dpix_loc, obj_num1);

    cudaMemcpy(pixels, dpixels, n*sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(dpixels);
    cudaFree(dpix_loc);
    cudaFree(dobj_type);
    cudaFree(dobj_pos);
    cudaFree(dobj_param1);
    cudaFree(dobj_rot);
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
