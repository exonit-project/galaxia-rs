// SPDX-License-Identifier: AGPL-3.0-or-later
// CUDA kernel for Galaxia 0.7.2 star generation (F4 GPU Acceleration)
// Accelerates the spatial rejection sampling, kinematics, and survey cuts.

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PHILOX_M4X32_0 0xD2511F53u
#define PHILOX_M4X32_1 0xCD9E8D57u
#define PHILOX_W32_0   0x9E3779B9u
#define PHILOX_W32_1   0xBB67AE85u

struct CudaActiveLeaf {
    double pos[4];      // x, y, z, age
    double l;           // half box length (kpc)
    double lage;        // half age length (yr)
    double rho_max;
    double rho_min;
    double dmod;
    double box_mmin;
    double box_mmax;
    double cpd_min;
    double cpd_max;
    uint32_t orig_leaf_idx;
    uint32_t n_stars;
    uint32_t star_offset;
    uint32_t pad;
};

struct CudaPopParams {
    int32_t pop_id;
    int32_t option_e;
    int32_t warp_flare_on;
    int32_t pad0;
    double age;
    double dage;
    double feh;
    double dfeh;
    double sigma_v[3];
    double pad1;
    double rho0;
    double hr1;
    double hr2;
    double xl;
    double r_warp;
    double r_flare;
    double g_warp;
    double g_flare;
    double th_max;
    double qparam;
    double bparam;
    double ac;
    double rhoc;
    double eps;
    double x0;
    double y0;
    double z0;
    double rc_bulge;
    double rot_matrix[3][3];
    double agevec[7];
    double fehvec[7];
};

struct CudaSurveyParams {
    double pos_c[6];
    double cone_dir[3];
    double cone_dth;
    double app_mag_limits[2];
    double color_limits[2];
    double abs_mag_limits[2];
    double r_max;
    uint64_t seed;
    uint32_t field_id;
    uint32_t pad;
};

struct CudaOutputStar {
    double pos[3];
    double vel[3];
    double age;
    double feh;
    double smass;
    double mags[3];
    double abs_mags[3];
    double rad;
    double dmod;
    int32_t popid;
    int32_t accepted;
};

// ============================================================================
// Philox4x32 PRNG Implementation matching src/rng.rs
// ============================================================================

__device__ __forceinline__ uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

struct PhiloxState {
    uint32_t key[2];
    uint32_t ctr[4];
    uint32_t buffer[4];
    int buffer_idx;
    double normal_cache;
    int has_normal_cache;
};

__device__ __forceinline__ void philox4x32_round(uint32_t* ctr, const uint32_t* key) {
    uint64_t prod0 = (uint64_t)ctr[0] * (uint64_t)PHILOX_M4X32_0;
    uint32_t hi0 = (uint32_t)(prod0 >> 32);
    uint32_t lo0 = (uint32_t)prod0;

    uint64_t prod1 = (uint64_t)ctr[2] * (uint64_t)PHILOX_M4X32_1;
    uint32_t hi1 = (uint32_t)(prod1 >> 32);
    uint32_t lo1 = (uint32_t)prod1;

    uint32_t c0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t c1 = lo1;
    uint32_t c2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t c3 = lo0;

    ctr[0] = c0;
    ctr[1] = c1;
    ctr[2] = c2;
    ctr[3] = c3;
}

__device__ __forceinline__ void philox4x32_10(const uint32_t* in_ctr, const uint32_t* in_key, uint32_t* out) {
    uint32_t ctr[4];
    ctr[0] = in_ctr[0]; ctr[1] = in_ctr[1]; ctr[2] = in_ctr[2]; ctr[3] = in_ctr[3];
    uint32_t key[2];
    key[0] = in_key[0]; key[1] = in_key[1];

    philox4x32_round(ctr, key);
    for (int i = 1; i < 10; ++i) {
        key[0] += PHILOX_W32_0;
        key[1] += PHILOX_W32_1;
        philox4x32_round(ctr, key);
    }
    out[0] = ctr[0]; out[1] = ctr[1]; out[2] = ctr[2]; out[3] = ctr[3];
}

__device__ void philox_init(PhiloxState* s, uint64_t seed, uint32_t field, uint32_t pop, uint32_t leaf, uint32_t star, uint32_t purpose) {
    s->key[0] = ((uint32_t)seed) ^ fmix32(purpose);
    s->key[1] = ((uint32_t)(seed >> 32)) ^ fmix32(pop);
    s->ctr[0] = 0;
    s->ctr[1] = star;
    s->ctr[2] = leaf;
    s->ctr[3] = field;
    philox4x32_10(s->ctr, s->key, s->buffer);
    s->buffer_idx = 0;
    s->has_normal_cache = 0;
    s->normal_cache = 0.0;
}

__device__ void philox_refill(PhiloxState* s) {
    s->ctr[0]++;
    if (s->ctr[0] == 0) {
        s->ctr[1]++;
        if (s->ctr[1] == 0) {
            s->ctr[2]++;
            if (s->ctr[2] == 0) {
                s->ctr[3]++;
            }
        }
    }
    philox4x32_10(s->ctr, s->key, s->buffer);
    s->buffer_idx = 0;
}

__device__ __forceinline__ uint32_t philox_next_u32(PhiloxState* s) {
    if (s->buffer_idx >= 4) {
        philox_refill(s);
    }
    return s->buffer[s->buffer_idx++];
}

__device__ __forceinline__ double philox_uniform_f64(PhiloxState* s) {
    uint32_t w0 = philox_next_u32(s);
    uint32_t w1 = philox_next_u32(s);
    uint64_t u53 = (((uint64_t)w0) << 21) | (((uint64_t)w1) >> 11);
    return ((double)u53 + 0.5) * (1.0 / 9007199254740992.0);
}

__device__ double philox_standard_normal(PhiloxState* s) {
    if (s->has_normal_cache) {
        s->has_normal_cache = 0;
        return s->normal_cache;
    }
    double u1 = philox_uniform_f64(s);
    double u2 = philox_uniform_f64(s);
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    double z0 = r * cos(theta);
    double z1 = r * sin(theta);
    s->normal_cache = z1;
    s->has_normal_cache = 1;
    return z0;
}

// ============================================================================
// Warp, Flare & Density Models
// ============================================================================

__device__ void thin_disk_set_pos_b(double x, double y, double z,
                                    double r_warp, double r_flare,
                                    double g_warp, double g_flare, double th_max,
                                    double* pos_b, double* k_flare) {
    double r = sqrt(x*x + y*y);
    double th = atan2(y, x);
    if (th < 0.0) th += 2.0 * M_PI;

    double zw = z;
    if (r > r_warp) {
        zw = z - g_warp * (r - r_warp) * cos(th - th_max);
    }

    double kf = (r > r_flare) ? 1.0 + g_flare * (r - r_flare) : 1.0;

    pos_b[0] = x;
    pos_b[1] = y;
    pos_b[2] = zw;
    *k_flare = kf;
}

__device__ __forceinline__ double thin_disk_eps(double age) {
    double val = 0.104 * sqrt((age + 0.1e9) / 10.1e9);
    return (val < 0.079) ? val : 0.079;
}

__device__ double thin_disk0_density(double x, double y, double z, double age, const CudaPopParams* pop) {
    double pos_b[3];
    double k_flare;
    thin_disk_set_pos_b(x, y, z, pop->r_warp, pop->r_flare, pop->g_warp, pop->g_flare, pop->th_max, pos_b, &k_flare);
    double eps1 = thin_disk_eps(age);
    double denom_z = eps1 * k_flare;
    double a2 = pos_b[0]*pos_b[0] + pos_b[1]*pos_b[1] + (pos_b[2]*pos_b[2]) / (denom_z * denom_z);
    return (pop->rho0 / eps1) * (exp(-a2 / pop->hr1) - exp(-a2 / pop->hr2)) / k_flare;
}

__device__ double thin_disk1_density(double x, double y, double z, double age, const CudaPopParams* pop) {
    double pos_b[3];
    double k_flare;
    thin_disk_set_pos_b(x, y, z, pop->r_warp, pop->r_flare, pop->g_warp, pop->g_flare, pop->th_max, pos_b, &k_flare);
    double eps1 = thin_disk_eps(age);
    double denom_z = eps1 * k_flare;
    double a2 = pos_b[0]*pos_b[0] + pos_b[1]*pos_b[1] + (pos_b[2]*pos_b[2]) / (denom_z * denom_z);
    double term1 = exp(-sqrt(0.25 + a2 / pop->hr1));
    double term2 = exp(-sqrt(0.25 + a2 / pop->hr2));
    return (pop->rho0 / eps1) * (term1 - term2) / k_flare;
}

__device__ double thick_disk_density(double x, double y, double z, const CudaPopParams* pop) {
    double pos_b[3];
    double k_flare;
    thin_disk_set_pos_b(x, y, z, pop->r_warp, pop->r_flare, pop->g_warp, pop->g_flare, pop->th_max, pos_b, &k_flare);
    double a2 = sqrt(pos_b[0]*pos_b[0] + pos_b[1]*pos_b[1]);
    double hr2 = pop->hr2;
    double xl = pop->xl;
    double temp = (2.0*hr2 + xl)*(3.0*k_flare*hr2*xl + 3.0*k_flare*k_flare*hr2*hr2 + xl*xl)
                  / ((2.0*hr2*k_flare + xl)*(3.0*hr2*xl + 3.0*hr2*hr2 + xl*xl));
    double radial_part = pop->rho0 * exp(-(a2 - 8.0) / pop->hr1);
    if (fabs(z) < xl) {
        return radial_part * (1.0 - pos_b[2]*pos_b[2] / (xl * (2.0*hr2*k_flare + xl))) / temp;
    } else {
        double z_scale = hr2 * k_flare;
        double factor = exp(xl / z_scale) / (1.0 + 0.5 * xl / z_scale);
        return radial_part * factor * exp(-fabs(pos_b[2]) / z_scale) / temp;
    }
}

__device__ double spheroid_density(double x, double y, double z, const CudaPopParams* pop) {
    double a = sqrt(x*x + y*y + (z*z) / (pop->eps * pop->eps));
    if (a < pop->ac) {
        return pop->rhoc;
    } else {
        return pop->rho0 * pow(a / 8.0, -2.44);
    }
}

__device__ double bulge_density(double x, double y, double z, const CudaPopParams* pop) {
    double bx = pop->rot_matrix[0][0]*x + pop->rot_matrix[0][1]*y + pop->rot_matrix[0][2]*z;
    double by = pop->rot_matrix[1][0]*x + pop->rot_matrix[1][1]*y + pop->rot_matrix[1][2]*z;
    double bz = pop->rot_matrix[2][0]*x + pop->rot_matrix[2][1]*y + pop->rot_matrix[2][2]*z;

    double term_xy = bx*bx / (pop->x0 * pop->x0) + by*by / (pop->y0 * pop->y0);
    double term_z = bz*bz / (pop->z0 * pop->z0);
    double a2 = sqrt(term_xy*term_xy + term_z*term_z);

    double cut = sqrt(bx*bx + by*by) - pop->rc_bulge;
    if (cut < 0.0) {
        return pop->rho0 * exp(-0.5 * a2);
    } else {
        return pop->rho0 * exp(-0.5 * a2) * exp(-cut*cut / 0.5);
    }
}

__device__ double eval_density(int pop_id, double x, double y, double z, double age, const CudaPopParams* pop) {
    if (pop_id == 0) {
        return thin_disk0_density(x, y, z, age, pop);
    } else if (pop_id >= 1 && pop_id <= 6) {
        return thin_disk1_density(x, y, z, age, pop);
    } else if (pop_id == 7) {
        return thick_disk_density(x, y, z, pop);
    } else if (pop_id == 8) {
        return spheroid_density(x, y, z, pop);
    } else if (pop_id == 9) {
        return bulge_density(x, y, z, pop);
    }
    return 0.0;
}

__device__ double thin_disk_dlnrho_dln_r(int pop_id, double x, double y, double z, double age, const CudaPopParams* pop) {
    double dr = 1e-3;
    double d2 = eval_density(pop_id, x * (1.0 + dr), y * (1.0 + dr), z, age, pop);
    double d1 = eval_density(pop_id, x * (1.0 - dr), y * (1.0 - dr), z, age, pop);
    return (d2 - d1) / (dr * (d1 + d2));
}

__device__ double eval_asymmetric_drift(int pop_id, double x, double y, double z, double vc, double age, double sigma_r, const CudaPopParams* pop) {
    double rc = sqrt(x*x + y*y);
    double sig0 = pop->sigma_v[0];
    double sig1 = pop->sigma_v[1];
    double sig2 = pop->sigma_v[2];

    if (pop_id <= 6) {
        double temp = (1.0 - sig1*sig1 / (sig0*sig0)) + (1.0 - sig2*sig2 / (sig0*sig0));
        temp += thin_disk_dlnrho_dln_r(pop_id, x, y, z, age, pop) - 2.0 * pop->qparam * rc / 2.5;
        temp *= sigma_r * sigma_r * 0.5 / vc;
        return -temp;
    } else if (pop_id == 7) {
        double temp = (1.0 - sig1*sig1 / (sig0*sig0)) + (1.0 - sig2*sig2 / (sig0*sig0));
        temp -= rc * (1.0 / pop->hr1 + 2.0 * pop->qparam / 2.5);
        temp *= sigma_r * sigma_r * 0.5 / vc;
        return -temp;
    } else if (pop_id == 8) {
        return vc; // (vc - va) == 0
    } else if (pop_id == 9) {
        return vc - rc * 71.62;
    }
    return 0.0;
}

__device__ double eval_sigmar(int pop_id, double age, double rc, const CudaPopParams* pop) {
    if (pop_id <= 6) {
        double val = 50.0 * pow((age + 0.1e9) / 10.1e9, pop->bparam);
        if (val > 43.1) val = 43.1;
        return val * exp((8.0 - rc) * pop->qparam / 2.5);
    } else {
        return pop->sigma_v[0];
    }
}

__device__ double eval_amr(int pop_id, double x, double y, double z, double age, const CudaPopParams* pop) {
    if (pop_id <= 6) {
        double r = sqrt(x*x + y*y);
        double feh_interp = pop->fehvec[0];
        if (age <= pop->agevec[0]) {
            feh_interp = pop->fehvec[0];
        } else if (age >= pop->agevec[6]) {
            feh_interp = pop->fehvec[6];
        } else {
            for (int i = 0; i < 6; ++i) {
                if (age >= pop->agevec[i] && age <= pop->agevec[i+1]) {
                    double frac = (age - pop->agevec[i]) / (pop->agevec[i+1] - pop->agevec[i]);
                    feh_interp = pop->fehvec[i] + frac * (pop->fehvec[i+1] - pop->fehvec[i]);
                    break;
                }
            }
        }
        return feh_interp - 0.07 * (r - 8.0);
    } else {
        return pop->feh;
    }
}

// ============================================================================
// Kinematics Coordinate Transforms
// ============================================================================

__device__ void vel_cylindrical_to_cartesian(double x, double y, double v_phi, double v_z, double v_r,
                                             double* vx, double* vy, double* vz) {
    double phi = atan2(y, x);
    double sin_phi = sin(phi);
    double cos_phi = cos(phi);
    *vx = -v_phi * sin_phi + v_r * cos_phi;
    *vy =  v_phi * cos_phi + v_r * sin_phi;
    *vz =  v_z;
}

__device__ void vel_spherical_to_cartesian(double x, double y, double z,
                                           double v_l, double v_b, double v_r,
                                           double* vx, double* vy, double* vz) {
    double r = sqrt(x*x + y*y + z*z);
    double l = atan2(y, x);
    if (l < 0.0) l += 2.0 * M_PI;
    double b = (r > 0.0) ? asin(fmax(-1.0, fmin(1.0, z / r))) : 0.0;

    double sin_l = sin(l);
    double cos_l = cos(l);
    double sin_b = sin(b);
    double cos_b = cos(b);

    *vx = -v_l * sin_l - v_b * sin_b * cos_l + v_r * cos_b * cos_l;
    *vy =  v_l * cos_l - v_b * sin_b * sin_l + v_r * cos_b * sin_l;
    *vz =  v_b * cos_b + v_r * sin_b;
}

// ============================================================================
// Tables: IMF and VCirc Interpolation
// ============================================================================

__device__ double sample_imf(const double* imf_x, const double* imf_cpd, uint32_t n, double target_cpd) {
    if (n < 2) return (n == 1) ? imf_x[0] : 0.0;
    if (target_cpd >= imf_cpd[0]) return imf_x[0];
    if (target_cpd <= imf_cpd[n - 1]) return imf_x[n - 1];

    uint32_t lo = 0;
    uint32_t hi = n - 1;
    while (hi - lo > 1) {
        uint32_t mid = (lo + hi) >> 1;
        if (imf_cpd[mid] >= target_cpd) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    double c0 = imf_cpd[lo];
    double c1 = imf_cpd[lo + 1];
    double x0 = imf_x[lo];
    double x1 = imf_x[lo + 1];
    double dc = c1 - c0;
    if (fabs(dc) < 1e-30) return x0;
    double frac = (target_cpd - c0) / dc;
    return x0 + frac * (x1 - x0);
}

__device__ double eval_vcirc(const double* vcirc_r, const double* vcirc_v, uint32_t n, double r) {
    if (n == 0) return 226.84;
    if (r <= vcirc_r[0]) return vcirc_v[0];
    if (r >= vcirc_r[n - 1]) return vcirc_v[n - 1];

    uint32_t lo = 0;
    uint32_t hi = n - 1;
    while (hi - lo > 1) {
        uint32_t mid = (lo + hi) >> 1;
        if (vcirc_r[mid] <= r) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    double r0 = vcirc_r[lo];
    double r1 = vcirc_r[lo + 1];
    double v0 = vcirc_v[lo];
    double v1 = vcirc_v[lo + 1];
    double dr = r1 - r0;
    if (fabs(dr) < 1e-30) return v0;
    return v0 + (v1 - v0) * (r - r0) / dr;
}

__device__ int locate_nearest_feh(const double* feh_grid, uint32_t n, double x1) {
    if (n == 0) return 0;
    if (n == 1 || x1 <= feh_grid[0]) return 0;
    if (x1 >= feh_grid[n - 1]) return n - 1;

    uint32_t lo = 0;
    uint32_t hi = n - 1;
    while (hi - lo > 1) {
        uint32_t mid = (lo + hi) >> 1;
        if (feh_grid[mid] <= x1) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    double frac = (x1 - feh_grid[lo]) / (feh_grid[lo + 1] - feh_grid[lo]);
    return (frac < 0.5) ? lo : (lo + 1);
}

// ============================================================================
// Main Star Spawning Kernel
// ============================================================================

extern "C" __global__ void spawn_kernel(
    CudaPopParams pop,
    CudaSurveyParams survey,
    const CudaActiveLeaf* __restrict__ leaves,
    const uint32_t* __restrict__ star_to_leaf,
    const uint32_t* __restrict__ star_in_leaf,
    const double* __restrict__ imf_x,
    const double* __restrict__ imf_cpd,
    uint32_t imf_size,
    const double* __restrict__ isochrone_m,
    const double* __restrict__ isochrone_mag0,
    const double* __restrict__ isochrone_mag1,
    const double* __restrict__ isochrone_mag2,
    const uint32_t* __restrict__ track_offsets,
    const double* __restrict__ track_m_min,
    const double* __restrict__ track_m_max,
    uint32_t num_ages,
    uint32_t num_feh,
    double age_grid_min,
    double d_age,
    const double* __restrict__ feh_grid,
    const double* __restrict__ vcirc_r,
    const double* __restrict__ vcirc_v,
    uint32_t vcirc_size,
    CudaOutputStar* __restrict__ out_stars,
    uint32_t total_stars
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_stars) return;

    out_stars[tid].accepted = 0;

    uint32_t leaf_idx = star_to_leaf[tid];
    uint32_t star_idx = star_in_leaf[tid];
    const CudaActiveLeaf leaf = leaves[leaf_idx];

    // Seed Philox RNG for this specific star
    PhiloxState rng;
    philox_init(&rng, survey.seed, survey.field_id, pop.pop_id, leaf.orig_leaf_idx, star_idx, 1);

    double x = 0.0, y = 0.0, z = 0.0;
    double star_age = 0.0;
    double age_linear = 0.0;

    if (pop.option_e == 1) {
        // Rejection sampling for position inside leaf box
        for (int iter = 0; iter < 100000; ++iter) {
            double u1 = philox_uniform_f64(&rng);
            double u2 = philox_uniform_f64(&rng);
            double u3 = philox_uniform_f64(&rng);
            x = leaf.pos[0] + (u1 - 0.5) * 2.0 * leaf.l;
            y = leaf.pos[1] + (u2 - 0.5) * 2.0 * leaf.l;
            z = leaf.pos[2] + (u3 - 0.5) * 2.0 * leaf.l;
            double u4 = philox_uniform_f64(&rng);
            star_age = leaf.pos[3] + (u4 - 0.5) * 2.0 * leaf.lage;

            double rho = eval_density(pop.pop_id, x, y, z, star_age, &pop);
            double u5 = philox_uniform_f64(&rng);
            if (u5 * (2.0 * leaf.rho_max) <= rho) {
                break;
            }
        }
        age_linear = star_age;
    } else {
        double u_age = philox_uniform_f64(&rng);
        age_linear = pop.age + (u_age - 0.5) * 2.0 * pop.dage;
    }

    double z0 = philox_standard_normal(&rng);
    double feh = eval_amr(pop.pop_id, x, y, z, age_linear, &pop) + z0 * pop.dfeh;

    double u_imf = philox_uniform_f64(&rng);
    double target_cpd = leaf.cpd_min + u_imf * (leaf.cpd_max - leaf.cpd_min);
    double smass = sample_imf(imf_x, imf_cpd, imf_size, target_cpd);
    double log10_age = log10(age_linear);

    // Isochrone interpolation
    int i_age = (int)floor((log10_age - age_grid_min) / d_age + 0.5);
    if (i_age < 0) i_age = 0;
    if (i_age >= (int)num_ages) i_age = num_ages - 1;

    int j_feh = locate_nearest_feh(feh_grid, num_feh, feh);
    uint32_t track_idx = j_feh * num_ages + i_age;

    uint32_t track_start = track_offsets[track_idx];
    uint32_t track_count = track_offsets[track_idx + 1] - track_start;
    double m_min = track_m_min[track_idx];
    double m_max = track_m_max[track_idx];

    double abs_mags[3] = { 1e6, 1e6, 1e6 };
    if (smass > m_min && smass < m_max && track_count >= 2) {
        uint32_t lo = track_start;
        uint32_t hi = track_start + track_count - 1;
        while (hi - lo > 1) {
            uint32_t mid = (lo + hi) >> 1;
            if (isochrone_m[mid] <= smass) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        double m0 = isochrone_m[lo];
        double m1 = isochrone_m[lo + 1];
        double dm = m1 - m0;
        double frac = (fabs(dm) > 1e-30) ? (smass - m0) / dm : 0.0;
        abs_mags[0] = isochrone_mag0[lo] + frac * (isochrone_mag0[lo + 1] - isochrone_mag0[lo]);
        abs_mags[1] = isochrone_mag1[lo] + frac * (isochrone_mag1[lo + 1] - isochrone_mag1[lo]);
        abs_mags[2] = isochrone_mag2[lo] + frac * (isochrone_mag2[lo + 1] - isochrone_mag2[lo]);
    }

    if (abs_mags[0] >= 1e4) {
        return; // Unphysical or below track limit
    }

    if (pop.option_e == 0) {
        // Rejection sampling for position occurs here for option_e == 0
        for (int iter = 0; iter < 100000; ++iter) {
            double u1 = philox_uniform_f64(&rng);
            double u2 = philox_uniform_f64(&rng);
            double u3 = philox_uniform_f64(&rng);
            x = leaf.pos[0] + (u1 - 0.5) * 2.0 * leaf.l;
            y = leaf.pos[1] + (u2 - 0.5) * 2.0 * leaf.l;
            z = leaf.pos[2] + (u3 - 0.5) * 2.0 * leaf.l;

            double rho = eval_density(pop.pop_id, x, y, z, pop.age, &pop);
            double u5 = philox_uniform_f64(&rng);
            if (u5 * (2.0 * leaf.rho_max) <= rho) {
                break;
            }
        }
    }

    // Kinematics
    double age_temp = pow(10.0, log10_age);
    double rc = sqrt(x*x + y*y);
    double vc = eval_vcirc(vcirc_r, vcirc_v, vcirc_size, rc);
    double temp = eval_sigmar(pop.pop_id, age_temp, rc, &pop);

    double z1 = philox_standard_normal(&rng);
    double z2 = philox_standard_normal(&rng);
    double z3 = philox_standard_normal(&rng);

    double sig_u = pop.sigma_v[0];
    double sig_v = pop.sigma_v[1];
    double sig_w = pop.sigma_v[2];

    double va = eval_asymmetric_drift(pop.pop_id, x, y, z, vc, age_temp, temp, &pop);

    double v3 = z1 * temp * sig_v / sig_u - (vc - va);
    double v4 = z2 * temp * sig_w / sig_u;
    double v5 = z3 * temp;

    double vx, vy, vz;
    if (pop.pop_id == 8) {
        vel_spherical_to_cartesian(x, y, z, v3, v4, v5, &vx, &vy, &vz);
    } else {
        vel_cylindrical_to_cartesian(x, y, v3, v4, v5, &vx, &vy, &vz);
    }

    // Shift to Heliocentric frame
    double hx = x - survey.pos_c[0];
    double hy = y - survey.pos_c[1];
    double hz = z - survey.pos_c[2];

    double hvx = vx - survey.pos_c[3];
    double hvy = vy - survey.pos_c[4];
    double hvz = vz - survey.pos_c[5];

    double rad = sqrt(hx*hx + hy*hy + hz*hz);
    double dmod = 5.0 * log10(rad * 100.0);

    double mags[3];
    mags[0] = abs_mags[0] + dmod;
    mags[1] = abs_mags[1] + dmod;
    mags[2] = abs_mags[2] + dmod;

    // Survey cuts
    if (mags[0] < survey.app_mag_limits[0] || mags[0] > survey.app_mag_limits[1]) return;
    if ((mags[1] - mags[2]) < survey.color_limits[0] || (mags[1] - mags[2]) > survey.color_limits[1]) return;
    if ((mags[0] - dmod) < survey.abs_mag_limits[0] || (mags[0] - dmod) > survey.abs_mag_limits[1]) return;
    if (rad > survey.r_max) return;

    // Observer patch cone check
    if (rad > 0.0) {
        double dot = (hx * survey.cone_dir[0] + hy * survey.cone_dir[1] + hz * survey.cone_dir[2]) / rad;
        if (dot > 1.0) dot = 1.0;
        if (dot < -1.0) dot = -1.0;
        if (acos(dot) >= survey.cone_dth) return;
    }

    // Accepted star particle!
    out_stars[tid].pos[0] = hx;
    out_stars[tid].pos[1] = hy;
    out_stars[tid].pos[2] = hz;

    out_stars[tid].vel[0] = hvx;
    out_stars[tid].vel[1] = hvy;
    out_stars[tid].vel[2] = hvz;

    out_stars[tid].age = log10_age;
    out_stars[tid].feh = feh;
    out_stars[tid].smass = smass;

    out_stars[tid].mags[0] = mags[0];
    out_stars[tid].mags[1] = mags[1];
    out_stars[tid].mags[2] = mags[2];

    out_stars[tid].abs_mags[0] = abs_mags[0];
    out_stars[tid].abs_mags[1] = abs_mags[1];
    out_stars[tid].abs_mags[2] = abs_mags[2];

    out_stars[tid].rad = rad;
    out_stars[tid].dmod = dmod;
    out_stars[tid].popid = pop.pop_id;
    out_stars[tid].accepted = 1;
}
