#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cassert>
#include <math.h>
#include <vector>

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
// Transparent hits split one ray into a reflected + refracted child, so the
// walk is a binary tree, not a chain -- traced with an explicit worklist
// stack (bounded, no device recursion) instead of the old single-ray loop.
#define MAX_STACK 32
#define MIN_RAY_WEIGHT 0.01f
#define MAX_SHADOW_HOPS 4

// Photon mapping pass for caustics: a separate light-side trace (see
// photon_kernel below) that actually bends rays through glass via Snell's
// law on the way FROM the light, which is the piece a camera-side ray tracer
// structurally cannot produce (see the light-transport discussion this is
// implementing -- an L S+ D path). Photons are only stored on the first
// opaque/diffuse surface reached after >=1 transparent bounce (a caustic
// map only, not a full global-illumination photon map), then gathered with
// a brute-force radius search at shading time -- consistent with the rest
// of this renderer, which has no acceleration structure for anything else
// either.
#define MAX_PHOTONS 20000
#define NUM_EMIT_PHOTONS 60000
#define PHOTON_MAX_DEPTH 8
#define CAUSTIC_GATHER_RADIUS 0.25f
#define CAUSTIC_BRIGHTNESS 400.0f

struct RayTask {
    float3 origin;
    float3 direction;
    // per-channel throughput, not a scalar -- refraction tints it by the
    // object's own diffuse color (colored glass), so different channels can
    // attenuate differently as a ray passes through one or more surfaces
    float3 weight;
    int depth;
};

inline __device__ float max_channel(float3& v) {
    return fmaxf(fmaxf(v.x, v.y), v.z);
}

// Returns the nearest root ahead of the ray (> epsilon), same convention as
// intersect_plane/intersect_box. The old version only handled both roots
// positive, i.e. origin outside the sphere -- refraction needs to trace a
// ray whose origin is INSIDE the sphere (looking for the exit point), where
// exactly one root is negative; that case used to silently return MISS.
inline __device__ float intersect_sphere(float3& center, float& size, float3& origin, float3& direction) {
    float3 tmp = origin-center;
    float b = 2*dot(direction, tmp);
    tmp = origin-center;
    float c = pow(norm_(tmp), 2) - pow(size, 2);
    float delta = pow(b, 2) - 4*c;
    if (delta > 0) {
        float sqrt_delta = sqrt(delta);
        float t1 = (-b-sqrt_delta)/2;
        float t2 = (-b+sqrt_delta)/2;
        if (t1 > 1e-4f) {
            return t1;
        }
        if (t2 > 1e-4f) {
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

// Dielectric split for transparency: Fresnel reflectance (Schlick's
// approximation) plus the Snell's-law refraction direction, in one pass
// since both need the same entering-vs-exiting bookkeeping. `ior` is the
// object's index of refraction; the far side is always assumed to be vacuum
// (1.0) -- there's no nested-medium tracking. Whether the ray is entering or
// exiting is read off the sign of dot(I, N): surface_normal's sphere/box
// normals are always the true outward normal, so this sign is reliable on
// both sides of the surface. On total internal reflection, refract_dir is
// zero and reflectance is forced to 1 so all remaining energy reflects.
inline __device__ void dielectric_split(float3& I, float3& N, float ior, float3& refract_dir, float& reflectance, int& total_internal_reflection) {
    float cosi = dot(I, N);
    float etai = 1.f;
    float etat = ior;
    float3 n = N;
    if (cosi < 0.f) {
        cosi = -cosi;
    } else {
        float tmp_eta = etai; etai = etat; etat = tmp_eta;
        n = flip(N);
    }
    float eta = etai/etat;
    float k = 1.f - eta*eta*(1.f-cosi*cosi);
    if (k < 0.f) {
        total_internal_reflection = 1;
        reflectance = 1.f;
        refract_dir = make_float3(0.f, 0.f, 0.f);
        return;
    }
    total_internal_reflection = 0;
    float3 scaled_i = eta*I;
    float coef = eta*cosi - sqrtf(k);
    float3 scaled_n = coef*n;
    float3 dir = scaled_i+scaled_n;
    refract_dir = normalize(dir);

    float r0 = (etai-etat)/(etai+etat);
    r0 = r0*r0;
    float x = 1.f-cosi;
    float x2 = x*x;
    float x5 = x2*x2*x;
    reflectance = r0 + (1.f-r0)*x5;
}

// Cheap deterministic hash-based PRNG (no curand dependency) -- seeded per
// photon/bounce/component index, not stateful, so it's trivially parallel.
inline __device__ float photon_rand(unsigned int seed) {
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15);
    return (float)(seed & 0x00FFFFFFu) / (float)0x01000000u;
}

// Uniform-solid-angle direction within a cone of half-angle acos(cos_max)
// around `axis`. Used to aim emitted photons at a transparent object instead
// of over the full 4*pi sphere -- otherwise, for a small glass object far
// from the light, only a tiny fraction of uniformly-emitted photons would
// ever hit it and almost none would land in the photon map at all.
inline __device__ float3 sample_cone_direction(float3& axis, float cos_max, float u1, float u2) {
    float cos_theta = 1.f - u1*(1.f-cos_max);
    float sin_theta = sqrtf(fmaxf(0.f, 1.f-cos_theta*cos_theta));
    float pi2 = 6.2831853f;
    float phi = pi2*u2;
    float3 up_ref = fabsf(axis.y) < 0.99f ? make_float3(0.f, 1.f, 0.f) : make_float3(1.f, 0.f, 0.f);
    float3 tangent1 = cross(up_ref, axis);
    tangent1 = normalize(tangent1);
    float3 tangent2 = cross(axis, tangent1);
    float cx = sin_theta*cosf(phi);
    float cy = sin_theta*sinf(phi);
    float3 t1_scaled = cx*tangent1;
    float3 t2_scaled = cy*tangent2;
    float3 axis_scaled = cos_theta*axis;
    float3 dir = t1_scaled+t2_scaled;
    dir = dir+axis_scaled;
    return normalize(dir);
}

// Brute-force radius search over the stored caustic photon map -- a flat
// disc density estimate (sum of nearby photon power / disc area), the
// simplest of the standard photon-mapping reconstruction kernels. No
// acceleration structure, matching the rest of this renderer's O(n) object
// loops; kept affordable by capping MAX_PHOTONS rather than by any spatial
// indexing.
inline __device__ float3 gather_caustics(float3 *photon_pos, float3 *photon_power, int num_photons, float3& point, float radius) {
    float3 sum = make_float3(0.f, 0.f, 0.f);
    float r2 = radius*radius;
    for (int p = 0; p < num_photons; ++p) {
        float3 d = point-photon_pos[p];
        float dist2 = d.x*d.x+d.y*d.y+d.z*d.z;
        if (dist2 < r2) {
            sum += photon_power[p];
        }
    }
    float pi = 3.14159265f;
    float area = pi*r2;
    float inv_area = 1.f/area;
    return inv_area*sum;
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

// Emits one photon per thread from the light, aimed (via sample_cone_direction)
// at one of the scene's transparent objects, and traces it forward: at each
// dielectric hit it reuses dielectric_split (the exact same Fresnel/Snell's
// law math the camera rays use) and stochastically picks ONE of reflect or
// refract, weighted by the Fresnel reflectance -- photons are discrete
// packets, so the split that camera rays do deterministically (spawning
// both children) is instead a Russian-roulette choice here, which is the
// standard photon-mapping approach and avoids an exponential blowup of paths
// per photon. The photon terminates and is stored the moment it hits a
// non-transparent (opaque) surface -- but ONLY if it passed through at least
// one dielectric bounce first (specular_bounces > 0), since a photon that
// went straight from light to a diffuse surface is already accounted for by
// the ordinary direct-lighting shadow-ray term; storing it too would double
// -count that light. This makes it a caustic-only photon map, not a full
// global-illumination one -- photons don't bounce further off opaque
// surfaces.
__global__
void photon_kernel(float3 *dphotons_pos, float3 *dphotons_power, int *dphoton_count,
 int *dobj_type, float3 *dobj_pos, float3 *dobj_param1, float4 *dobj_rot, float *dobj_size,
 float3 *dobj_diff, float *dobj_transparency, float *dobj_ior, float3 *dlights,
 int *dtarget_idx, float *dtarget_radius, int num_targets, int num_obj, int num_emit) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_emit || num_targets == 0) {
        return;
    }

    float3 light = dlights[0];
    int target = dtarget_idx[idx % num_targets];
    float3 target_center = dobj_pos[target];
    float target_radius = dtarget_radius[idx % num_targets];

    float3 to_target = target_center-light;
    float dist_to_target = norm_(to_target);
    float3 axis = normalize(to_target);

    // pad the cone a bit past the object's exact silhouette so grazing-angle
    // refractions near the limb aren't clipped out of the sample set
    float safe_radius = 1.15f*target_radius;
    float sin_max = safe_radius/dist_to_target;
    sin_max = fminf(sin_max, 1.f);
    float cos_max = sqrtf(fmaxf(0.f, 1.f-sin_max*sin_max));
    float pi = 3.14159265f;
    float solid_angle = 2.f*pi*(1.f-cos_max);

    float u1 = photon_rand((unsigned int)(idx*2u+1u));
    float u2 = photon_rand((unsigned int)(idx*2u+2u));
    float3 direction = sample_cone_direction(axis, cos_max, u1, u2);

    float3 origin = light;
    // importance-sampling correction: only ~num_emit/num_targets photons
    // are actually sent into this target's cone (not num_emit total, since
    // the budget is round-robined across every target), so the "fair share"
    // per-photon power has to be computed against that smaller per-target
    // count -- otherwise the total power delivered to each target ends up
    // short by a factor of num_targets. It's then scaled by
    // (solid_angle / 4*pi), the fraction of directions a uniformly-emitted
    // budget would have sent this way, so the total energy reaching this
    // target stays calibrated the same regardless of how concentrated (or
    // how many targets share) the cone sampling is.
    float full_sphere = 4.f*pi;
    float importance = solid_angle/full_sphere;
    float photons_per_target = (float)num_emit/(float)num_targets;
    float per_photon = CAUSTIC_BRIGHTNESS/photons_per_target;
    float power_scale = per_photon*importance;
    float3 power = make_float3(power_scale, power_scale, power_scale);

    int specular_bounces = 0;
    float shift = 0.00001f;

    for (int depth = 0; depth < PHOTON_MAX_DEPTH; ++depth) {
        float2 res = nearest_intersected_object(dobj_type, dobj_pos, dobj_param1, dobj_rot, dobj_size, origin, direction, num_obj);
        int hit = res.x;
        if (hit > num_obj) {
            return;
        }
        float3 tmp = res.y*direction;
        float3 hit_point = origin+tmp;
        float3 normal = surface_normal(dobj_type[hit], dobj_pos[hit], dobj_param1[hit], dobj_rot[hit], hit_point, direction);
        float transparency = dobj_transparency[hit];

        if (transparency <= 0.f) {
            if (specular_bounces > 0) {
                int slot = atomicAdd(dphoton_count, 1);
                if (slot < MAX_PHOTONS) {
                    dphotons_pos[slot] = hit_point;
                    dphotons_power[slot] = power;
                }
            }
            return;
        }

        specular_bounces++;
        float3 refract_dir;
        float reflectance;
        int tir;
        dielectric_split(direction, normal, dobj_ior[hit], refract_dir, reflectance, tir);
        float u3 = photon_rand((unsigned int)(idx*131u+depth*977u+3u));
        if (tir || u3 < reflectance) {
            float3 reflect_dir = reflect(direction, normal);
            tmp = shift*normal;
            origin = hit_point+tmp;
            direction = reflect_dir;
        } else {
            float3 tint = dobj_diff[hit];
            power = tint*power;
            float side = dot(refract_dir, normal) < 0.f ? -shift : shift;
            tmp = side*normal;
            origin = hit_point+tmp;
            direction = refract_dir;
        }
    }
}

__global__
void rt_kernel(float3 *dpixels, float *dobj_size, float *dobj_shine, float *dobj_refl,
 int *dobj_type, float3 *dobj_pos, float3 *dobj_param1, float4 *dobj_rot, float3 *dobj_amb, float3 *dobj_diff, float3 *dobj_spec,
  float3 *dcameras, float3 *dlights, float2 *dpix_loc, int num_obj, float *dobj_transparency, float *dobj_ior,
  float3 *dphotons_pos, float3 *dphotons_power, int num_photons) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    float3 amb_l = {1, 1, 1};
    float3 diff_l = {1, 1, 1};
    float3 spec_l = {1, 1, 1};

    float3 light = dlights[0];
    float3 tmp = {0, 0, 0};

    //camera is a (position, forward, up) basis: right/up are re-derived each
    //frame so interpolated forward/up keyframes stay orthonormal
    float3 eye = dcameras[0];
    float3 forward = normalize(dcameras[1]);
    float3 up_raw = dcameras[2];
    float3 right_raw = cross(forward, up_raw);
    float3 right = normalize(right_raw);
    float3 up = cross(right, forward);

    //the screen is tied to the camera basis
    float3 screen_x = dpix_loc[i].x*right;
    float3 screen_y = dpix_loc[i].y*up;
    tmp = forward+screen_x;
    float3 primary_direction = tmp+screen_y;
    primary_direction = normalize(primary_direction);

    // Glass needs entry hit + exit hit + whatever's behind it = 3 levels
    // minimum to see through at all, plus one spare for an extra bounce
    // (e.g. reflections off the entry face, or a mirror behind the glass).
    int max_depth = 4;
    float shift = 0.00001f;
    float3 color = {0, 0, 0};

    RayTask stack[MAX_STACK];
    int sp = 0;
    stack[0].origin = eye;
    stack[0].direction = primary_direction;
    stack[0].weight = make_float3(1.f, 1.f, 1.f);
    stack[0].depth = 0;
    sp = 1;

    while (sp > 0) {
        sp--;
        float3 origin = stack[sp].origin;
        float3 direction = stack[sp].direction;
        float3 weight = stack[sp].weight;
        int depth = stack[sp].depth;

        float2 res = nearest_intersected_object(dobj_type, dobj_pos, dobj_param1, dobj_rot, dobj_size, origin, direction, num_obj);
        int nearest_obj = res.x;
        if (nearest_obj > num_obj) {
            continue;
        }
        tmp = res.y*direction;
        float3 intersection = origin+tmp;
        float3 normal_to_surface = surface_normal(dobj_type[nearest_obj], dobj_pos[nearest_obj], dobj_param1[nearest_obj], dobj_rot[nearest_obj], intersection, direction);

        //cheap procedural checker for spheres: obj_param1.x doubles as a
        //"checkered" flag on spheres (otherwise unused for that type). Split
        //the sphere's own rotated local frame into octants by axis sign and
        //XOR-parity them into two shades -- no UV unwrap, no pole pinch, and
        //it reuses the same quat rotation already carried for boxes, so a
        //rolling sphere (dobj_rot animated by rolling-without-slipping in
        //scene_io.py / scene-model.js) visibly spins the pattern.
        float checker_mask = 1.f;
        if (dobj_type[nearest_obj] == TYPE_SPHERE && dobj_param1[nearest_obj].x > 0.5f) {
            float3 center = dobj_pos[nearest_obj];
            float3 rel = intersection-center;
            float4 rotation = dobj_rot[nearest_obj];
            float4 inv_rot = quat_conj(rotation);
            float3 local = quat_rotate(rel, inv_rot);
            local = normalize(local);
            int sx = local.x > 0.f;
            int sy = local.y > 0.f;
            int sz = local.z > 0.f;
            int parity = sx ^ sy ^ sz;
            checker_mask = parity ? 1.f : 0.35f;
        }

        tmp = shift*normal_to_surface;
        float3 shifted_point = intersection+tmp;
        tmp = light-shifted_point;
        float3 intersection_to_light = normalize(tmp);
        tmp = light-intersection;
        float intersection_to_light_distance = norm_(tmp);

        // Shadow ray, walked through any transparent occluders instead of
        // stopping dead at the first hit -- otherwise a glass object casts a
        // fully opaque shadow (looks like solid black behind it) even though
        // it renders as see-through from the camera. Each transparent hop
        // multiplies the light's transmittance by that object's own
        // transparency*diffuse (so colored glass casts a tinted shadow, same
        // color logic as the refracted ray); an opaque hit blocks it
        // completely, same as before. The ray direction is NOT bent through
        // these hops (no Snell's law here) -- only its strength/color is
        // attenuated, which is the usual cheat for shadows through glass.
        float3 shadow_transmittance = make_float3(1.f, 1.f, 1.f);
        float3 shadow_origin = shifted_point;
        float shadow_remaining = intersection_to_light_distance;
        for (int hop = 0; hop < MAX_SHADOW_HOPS; ++hop) {
            float2 shadow_res = nearest_intersected_object(dobj_type, dobj_pos, dobj_param1, dobj_rot, dobj_size, shadow_origin, intersection_to_light, num_obj);
            if (shadow_res.y >= shadow_remaining) {
                break;
            }
            int occluder = shadow_res.x;
            float occ_transparency = dobj_transparency[occluder];
            if (occ_transparency <= 0.f) {
                shadow_transmittance = make_float3(0.f, 0.f, 0.f);
                break;
            }
            float3 occ_tint = dobj_diff[occluder];
            float3 occ_pass = occ_transparency*occ_tint;
            shadow_transmittance = occ_pass*shadow_transmittance;

            tmp = shadow_res.y*intersection_to_light;
            float3 occ_hit = shadow_origin+tmp;
            tmp = shift*intersection_to_light;
            shadow_origin = occ_hit+tmp;
            shadow_remaining = shadow_remaining-shadow_res.y;
        }
        int in_shadow = max_channel(shadow_transmittance) <= 0.f;

        float transparency = dobj_transparency[nearest_obj];
        float3 illumination = {0, 0, 0};

        if (!in_shadow) {
            //amb
            tmp = amb_l*dobj_amb[nearest_obj];
            tmp = checker_mask*tmp;
            tmp = shadow_transmittance*tmp;
            illumination += tmp;

            //diffuse
            float diff_dot = dot(intersection_to_light, normal_to_surface);
            tmp = diff_dot*diff_l;
            tmp = dobj_diff[nearest_obj]*tmp;
            tmp = checker_mask*tmp;
            tmp = shadow_transmittance*tmp;
            illumination += tmp;

            //specular
            tmp = eye-intersection;
            float3 intersection_to_camera = normalize(tmp);
            tmp = intersection_to_light+intersection_to_camera;
            float3 H = normalize(tmp);
            tmp = spec_l*dobj_spec[nearest_obj];
            float spec_pow = pow(dot(normal_to_surface, H), (dobj_shine[nearest_obj]/4));
            tmp = spec_pow*tmp;
            tmp = shadow_transmittance*tmp;
            illumination += tmp;
        }

        // Caustics: light that reached this point via a completely different
        // path (through a nearby glass object, recorded during the photon
        // pass) than the straight-line shadow ray above -- so it's added
        // unconditionally, not gated by in_shadow. A point can be dark by
        // the direct-light shadow test yet still catch focused, redirected
        // light this way, which is exactly the physical behavior direct
        // shadow rays alone can't produce.
        if (num_photons > 0) {
            float3 caustic = gather_caustics(dphotons_pos, dphotons_power, num_photons, intersection, CAUSTIC_GATHER_RADIUS);
            float3 diff_color = dobj_diff[nearest_obj];
            caustic = diff_color*caustic;
            caustic = checker_mask*caustic;
            illumination += caustic;
        }

        // transparency mutes how much of the surface's own Phong shading
        // shows through -- a fully clear object (transparency 1) skips
        // it almost entirely and is built up from reflection/refraction
        // instead
        float opacity_factor = 1.f-transparency;
        float3 local_weight = opacity_factor*weight;
        illumination = local_weight*illumination;
        color += illumination;

        //don't bother pushing children once they'd be too faint to matter,
        //past the depth budget, or (defensively) near the stack cap
        if (depth+1 >= max_depth || sp >= MAX_STACK-2) {
            continue;
        }

        if (transparency <= 0.f) {
            // opaque mirror-style reflection, same as before: stops (like
            // the original loop's break) once the surface itself is in
            // shadow, since there's nothing else driving this ray onward
            if (in_shadow) {
                continue;
            }
            float refl_factor = dobj_refl[nearest_obj];
            float3 refl_weight = refl_factor*weight;
            if (max_channel(refl_weight) > MIN_RAY_WEIGHT) {
                float3 reflect_dir = reflect(direction, normal_to_surface);
                stack[sp].origin = shifted_point;
                stack[sp].direction = reflect_dir;
                stack[sp].weight = refl_weight;
                stack[sp].depth = depth+1;
                sp++;
            }
        } else {
            // dielectric: reflect + refract split by Fresnel reflectance.
            // Unlike the opaque case this always continues even if the hit
            // point is in local shadow -- transmission through the object is
            // independent of whether its own surface can see the light.
            float3 refract_dir;
            float reflectance;
            int tir;
            dielectric_split(direction, normal_to_surface, dobj_ior[nearest_obj], refract_dir, reflectance, tir);

            // Fresnel reflection off glass is colorless (it's a surface
            // effect, light doesn't travel through the medium), so the
            // reflect branch is NOT tinted by the object's color -- only the
            // refract branch below is, since that's the light that actually
            // passes through the glass and picks up its color.
            float refl_factor = transparency*reflectance;
            float3 refl_weight = refl_factor*weight;
            if (max_channel(refl_weight) > MIN_RAY_WEIGHT) {
                float3 reflect_dir = reflect(direction, normal_to_surface);
                stack[sp].origin = shifted_point;
                stack[sp].direction = reflect_dir;
                stack[sp].weight = refl_weight;
                stack[sp].depth = depth+1;
                sp++;
            }

            if (!tir && sp < MAX_STACK) {
                // tint by the object's own diffuse color -- "the ball color"
                // as set in the inspector/scene JSON. White diffuse gives
                // clear (untinted) glass; a saturated diffuse color gives
                // colored glass, darkening further with each surface it's
                // refracted through (a cheap stand-in for Beer's-law
                // absorption, not physically exact but the right shape).
                float refr_factor = transparency*(1.f-reflectance);
                float3 refr_weight = refr_factor*weight;
                float3 tint = dobj_diff[nearest_obj];
                refr_weight = tint*refr_weight;
                if (max_channel(refr_weight) > MIN_RAY_WEIGHT) {
                    // offset along the transmitted ray's own side of the
                    // surface (not always +normal like the reflect/shadow
                    // bias above) so it doesn't immediately re-hit the same
                    // surface it just passed through
                    float side = dot(refract_dir, normal_to_surface) < 0.f ? -shift : shift;
                    tmp = side*normal_to_surface;
                    float3 refr_origin = intersection+tmp;
                    stack[sp].origin = refr_origin;
                    stack[sp].direction = refract_dir;
                    stack[sp].weight = refr_weight;
                    stack[sp].depth = depth+1;
                    sp++;
                }
            }
        }
    }

    //pixels is array of float need to be array of float3 to store rgb
    dpixels[i] = color;
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
                float *obj_transparency, int obj_num8,
                float *obj_ior, int obj_num9,
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
    assert(obj_num7==obj_num8);
    assert(obj_num8==obj_num9);
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
    float *dobj_size, *dobj_shine, *dobj_refl, *dobj_transparency, *dobj_ior;
    int *dobj_type;
    float2 *dpix_loc;

    cudaMalloc(&dobj_type, obj_num1*sizeof(int));
    cudaMalloc(&dobj_size, obj_num1*sizeof(float));
    cudaMalloc(&dobj_shine, obj_num1*sizeof(float));
    cudaMalloc(&dobj_refl, obj_num1*sizeof(float));
    cudaMalloc(&dobj_transparency, obj_num1*sizeof(float));
    cudaMalloc(&dobj_ior, obj_num1*sizeof(float));

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
    cudaMemcpy(dobj_transparency, obj_transparency, obj_num1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dobj_ior, obj_ior, obj_num1*sizeof(float), cudaMemcpyHostToDevice);

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

    // Photon pass: find which objects can actually cast a caustic (dielectric,
    // non-plane -- an infinite plane has no meaningful "as seen from the
    // light" cone to aim photons into) directly from the host-side arrays
    // already passed in, so this needs no new scene-JSON/Python plumbing at
    // all. Emission is aimed at each target's cone rather than uniformly over
    // 4*pi (see photon_kernel), since for a small glass object far from the
    // light almost none of a uniformly-emitted budget would ever hit it.
    vector<int> target_idx;
    vector<float> target_radius;
    for (int j = 0; j < obj_num1; ++j) {
        if (obj_transparency[j] <= 0.f || obj_type[j] == 1 /* TYPE_PLANE */) {
            continue;
        }
        float radius;
        if (obj_type[j] == 2 /* TYPE_BOX */) {
            float hx = obj_param1[j*3+0];
            float hy = obj_param1[j*3+1];
            float hz = obj_param1[j*3+2];
            radius = sqrtf(hx*hx+hy*hy+hz*hz);
        } else {
            radius = obj_size[j];
        }
        target_idx.push_back(j);
        target_radius.push_back(radius);
    }

    float3 *dphotons_pos = NULL;
    float3 *dphotons_power = NULL;
    int *dtarget_idx = NULL;
    float *dtarget_radius = NULL;
    int *dphoton_count = NULL;
    int num_photons_valid = 0;

    if (!target_idx.empty()) {
        cudaMalloc(&dphotons_pos, MAX_PHOTONS*sizeof(float3));
        cudaMalloc(&dphotons_power, MAX_PHOTONS*sizeof(float3));
        cudaMalloc(&dphoton_count, sizeof(int));
        cudaMemset(dphoton_count, 0, sizeof(int));
        cudaMalloc(&dtarget_idx, target_idx.size()*sizeof(int));
        cudaMalloc(&dtarget_radius, target_radius.size()*sizeof(float));
        cudaMemcpy(dtarget_idx, target_idx.data(), target_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dtarget_radius, target_radius.data(), target_radius.size()*sizeof(float), cudaMemcpyHostToDevice);

        int photon_threads = 256;
        int photon_blocks = (NUM_EMIT_PHOTONS+photon_threads-1)/photon_threads;
        photon_kernel<<<photon_blocks, photon_threads>>>(dphotons_pos, dphotons_power, dphoton_count,
         dobj_type, dobj_pos, dobj_param1, dobj_rot, dobj_size, dobj_diff, dobj_transparency, dobj_ior, dlights,
         dtarget_idx, dtarget_radius, (int)target_idx.size(), obj_num1, NUM_EMIT_PHOTONS);

        int photon_count_raw = 0;
        cudaMemcpy(&photon_count_raw, dphoton_count, sizeof(int), cudaMemcpyDeviceToHost);
        num_photons_valid = photon_count_raw < MAX_PHOTONS ? photon_count_raw : MAX_PHOTONS;
    }

    rt_kernel<<<n/512, 512>>>(dpixels, dobj_size, dobj_shine, dobj_refl,
     dobj_type, dobj_pos, dobj_param1, dobj_rot, dobj_amb, dobj_diff, dobj_spec, dcameras, dlights, dpix_loc, obj_num1,
     dobj_transparency, dobj_ior, dphotons_pos, dphotons_power, num_photons_valid);

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
    cudaFree(dobj_transparency);
    cudaFree(dobj_ior);
    if (dphotons_pos) cudaFree(dphotons_pos);
    if (dphotons_power) cudaFree(dphotons_power);
    if (dphoton_count) cudaFree(dphoton_count);
    if (dtarget_idx) cudaFree(dtarget_idx);
    if (dtarget_radius) cudaFree(dtarget_radius);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
            cudaGetErrorString(cudaerr));
}
