#include "test_helpers.hpp"

double mean(double *array,
            int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += array[i];

    return mean / num_elements;
}

double mean(double3 *array,
            int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += array[i].x + array[i].y + array[i].z;

    return mean / 3. / num_elements;
}

double mean_x(double3 *array,
              int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += array[i].x;

    return mean / num_elements;
}

double mean_y(double3 *array,
              int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += array[i].y;

    return mean / num_elements;
}

double mean_z(double3 *array,
              int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += array[i].z;

    return mean / num_elements;
}

double mean_norm(double3 *array,
                 int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += norm(array[i]);

    return mean / num_elements;
}

double mean_modified_radius(double3 *pos,
                            int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += sqrt(pos[i].x*pos[i].x +
                     pos[i].y*pos[i].y +
                     4*pos[i].z*pos[i].z);

    return mean / num_elements;
}

double std_dev(double *array,
               int num_elements) {
    double mu = mean(array,
                     num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i)
        sum += (array[i]-mu) * (array[i]-mu);

    return sqrt(sum / num_elements);
}

double std_dev(double3 *array,
               int num_elements) {
    double mu = mean(array,
                     num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i) {
        sum += (array[i].x-mu) * (array[i].x-mu) +
               (array[i].y-mu) * (array[i].y-mu) +
               (array[i].z-mu) * (array[i].z-mu);
    }

    return sqrt(sum / 3. / num_elements);
}

double std_dev_x(double3 *array,
                 int num_elements) {
    double mu = mean_x(array,
                       num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i) {
        sum += (array[i].x-mu) * (array[i].x-mu);
    }

    return sqrt(sum / num_elements);
}

double std_dev_y(double3 *array,
                 int num_elements) {
    double mu = mean_y(array,
                       num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i) {
        sum += (array[i].y-mu) * (array[i].y-mu);
    }

    return sqrt(sum / num_elements);
}

double std_dev_z(double3 *array,
                 int num_elements) {
    double mu = mean_z(array,
                       num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i) {
        sum += (array[i].z-mu) * (array[i].z-mu);
    }

    return sqrt(sum / num_elements);
}

double std_norm(double3 *vel,
                int num_elements) {
    double mu = mean_norm(vel,
                          num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i)
        sum += (norm(vel[i])-mu) * (norm(vel[i])-mu);

    return sqrt(sum / num_elements);
}

double std_modified_radius(double3 *pos,
                           int num_elements) {
    double mu = mean_modified_radius(pos,
                                     num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i)
        sum += (sqrt(pos[i].x*pos[i].x +
                     pos[i].y*pos[i].y +
                     4*pos[i].z*pos[i].z) - mu) *
               (sqrt(pos[i].x*pos[i].x +
                     pos[i].y*pos[i].y +
                     4*pos[i].z*pos[i].z) - mu);

    return sqrt(sum / num_elements);
}

double z_score(double value,
               double mean,
               double std) {
    return (value - mean) / std;
}

double mean_kinetic_energy(int num_atoms,
                           double3 *vel) {
    double total_kinetic_energy = 0.;

    for (int atom = 0; atom < num_atoms; ++atom) {
        total_kinetic_energy += kinetic_energy(vel[atom]);
    }

    return total_kinetic_energy / num_atoms;
}
