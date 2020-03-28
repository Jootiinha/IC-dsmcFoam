#include "define_device_constants.cuh"
#include "trapping_potential.cuh"

__host__ __device__ double3 B(double3 pos,
                               trap_geo params) {
    double3 mag_field = make_double3(0., 0., 0.);

    mag_field.x = 0.5 * params.Bz * pos.x;
    mag_field.y = 0.5 * params.Bz * pos.y;
    mag_field.z =-1.0 * params.Bz * pos.z;

    return mag_field;
}

__host__ __device__ double3 dB_dx(double3 pos,
                                  trap_geo params) {
    double3 dBdx = make_double3(0.5 * params.Bz,
                                0.,
                                0.);

    return dBdx;
}

__host__ __device__ double3 dB_dy(double3 pos,
                                  trap_geo params) {
    double3 dBdy = make_double3(0.,
                                0.5 * params.Bz,
                                0.);

    return dBdy;
}

__host__ __device__ double3 dB_dz(double3 pos,
                                  trap_geo params) {
    double3 dBdz = make_double3(0.,
                                0.,
                                -1.0 * params.Bz);

    return dBdz;
}

__device__ double d_dV_dx(double3 pos,
                          trap_geo params) {
    double3 unit_B = unit(B(pos,
                            params));
    return -0.5 * d_muB * d_gs * dot(unit_B,
                                     dB_dx(pos,
                                           params));
}

__device__ double d_dV_dy(double3 pos,
                          trap_geo params) {
    double3 unit_B = unit(B(pos,
                            params));
    return -0.5 * d_muB * d_gs * dot(unit_B,
                                     dB_dy(pos,
                                           params));
}

__device__ double d_dV_dz(double3 pos,
                          trap_geo params) {
    double3 unit_B = unit(B(pos,
                            params));
    return -0.5 * d_muB * d_gs * dot(unit_B,
                                     dB_dz(pos,
                                           params));
}
