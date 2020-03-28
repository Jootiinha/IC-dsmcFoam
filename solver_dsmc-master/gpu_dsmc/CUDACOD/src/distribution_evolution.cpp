
#include "distribution_evolution.hpp"
#ifdef CUDA
#include "distribution_evolution.cuh"
#endif

#include "declare_host_constants.hpp"

void velocity_verlet_update(int num_atoms,
                            double dt,
                            trap_geo params,
                            cublasHandle_t cublas_handle,
                            double3 *pos,
                            double3 *vel,
                            double3 *acc) {
    update_velocities(num_atoms,
                      0.5*dt,
                      cublas_handle,
                      acc,
                      vel);
    update_positions(num_atoms,
                     dt,
                     cublas_handle,
                     vel,
                     pos);
    update_accelerations(num_atoms,
                         params,
                         pos,
                         acc);
    update_velocities(num_atoms,
                      0.5*dt,
                      cublas_handle,
                      acc,
                      vel);
    return;
}

void sympletic_euler_update(int num_atoms,
                            double dt,
                            trap_geo params,
                            cublasHandle_t cublas_handle,
                            double3 *pos,
                            double3 *vel,
                            double3 *acc) {
    update_velocities(num_atoms,
                      dt,
                      cublas_handle,
                      acc,
                      vel);
    update_positions(num_atoms,
                     dt,
                     cublas_handle,
                     vel,
                     pos);
    update_accelerations(num_atoms,
                         params,
                         pos,
                         acc);

    return;
}

/** \fn void update_positions(int num_atoms,
 *                            double dt,
 *                            double3 *vel,
 *                            double3 *pos)  
 *  \brief Calls the function to update a `double3` host or device array with
 *  positions based on the atoms position, velocity and given time step.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param dt Length of the time step (seconds).
 *  \param *vel Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the velocities.
 *  \param *pos Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the positions.
 *  \exception not yet.
 *  \return void
*/

void update_positions(int num_atoms,
                      double dt,
                      cublasHandle_t cublas_handle,
                      double3 *vel,
                      double3 *pos) {
#if defined(CUDA)
    cu_update_positions(num_atoms,
                        dt,
                        cublas_handle,
                        vel,
                        pos);
#elif defined(MKL)
    cblas_daxpy(3*num_atoms,
                dt,
                reinterpret_cast<double *>(vel),
                1,
                reinterpret_cast<double *>(pos),
                1);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        pos[atom] = update_atom_position(dt,
                                         pos[atom],
                                         vel[atom]);
    }
#endif

    return;
}

double3 update_atom_position(double dt,
                             double3 pos,
                             double3 vel) {
    return pos + vel * dt;
}

/** \fn void update_velocities(int num_atoms,
 *                             double dt,
 *                             double3 *acc,
 *                             double3 *vel) 
 *  \brief Calls the function to update a `double3` host or device array with
 *  velocities based on the atoms velocity, acceleration and given time step.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param dt Length of the time step (seconds).
 *  \param *acc Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the accelerations.
 *  \param *vel Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the velocities.
 *  \exception not yet.
 *  \return void
*/

void update_velocities(int num_atoms,
                       double dt,
                       cublasHandle_t cublas_handle,
                       double3 *acc,
                       double3 *vel) {
#if defined(CUDA)
    cu_update_velocities(num_atoms,
                         dt,
                         cublas_handle,
                         acc,
                         vel);
#elif defined(MKL)
    cblas_daxpy(3*num_atoms,
                dt,
                reinterpret_cast<double *>(acc),
                1,
                reinterpret_cast<double *>(vel),
                1);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        vel[atom] = update_atom_velocity(dt,
                                         vel[atom],
                                         acc[atom]);
    }
#endif

    return;
}

double3 update_atom_velocity(double dt,
                             double3 vel,
                             double3 acc) {
    return vel + acc * dt;
}

/** \fn void update_accelerations(int num_atoms,
 *                                trap_geo params,
 *                                double3 *pos,
 *                                double3 *acc) 
 *  \brief Calls the function to update a `double3` host or device array with
 *  accelerations based on the atoms position and the trapping potential.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param params Customized structure of type `trap_geo` containing the 
 *  necessary constants for describing the trapping potential.
 *  \param *pos Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the positions.
 *  \param *acc Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the accelerations.
 *  \exception not yet.
 *  \return void
*/

void update_accelerations(int num_atoms,
                          trap_geo params,
                          double3 *pos,
                          double3 *acc) {
#ifdef CUDA
    cu_update_accelerations(num_atoms,
                            params,
                            pos,
                            acc);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        acc[atom] = update_atom_acceleration(params,
                                             pos[atom]);
    }
#endif

    return;
}

double3 update_atom_acceleration(trap_geo params,
                                 double3 pos) {
    double3 acc = make_double3(0., 0., 0.);

    acc.x = dV_dx(pos,
                  params) / mass;
    acc.y = dV_dy(pos,
                  params) / mass;
    acc.z = dV_dz(pos,
                  params) / mass;

    return acc;
}
