// SPDX-License-Identifier: AGPL-3.0-or-later
// OpenCL kernel for Galaxia 0.7.2 star generation
// Accelerates the spatial rejection sampling, kinematics, and survey cuts.
// Supports both strict FP64 (Intel Arc / NVIDIA / Apple / AMD ROCm) and FP32 (AMD Vega iGPU via Mesa rusticl).

#ifdef REAL_IS_FP32
#define REAL float
#define REAL_IS_FP64 0
#else
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
#define REAL double
#define REAL_IS_FP64 1
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PHILOX_M4X32_0 0xD2511F53u
#define PHILOX_M4X32_1 0xCD9E8D57u
#define PHILOX_W32_0   0x9E3779B9u
#define PHILOX_W32_1   0xBB67AE85u

typedef struct {
    REAL pos[4];      // x, y, z, age
    REAL l;           // half box length (kpc)
    REAL lage;        // half age length (yr)
    REAL rho_max;
    REAL rho_min;
    REAL dmod;
    REAL box_mmin;
    REAL box_mmax;
    REAL cpd_min;
    REAL cpd_max;
    uint orig_leaf_idx;
    uint n_stars;
    uint star_offset;
    uint pad;
} ClActiveLeaf;

typedef struct {
    int pop_id;
    int option_e;
    int warp_flare_on;
    int pad0;
    REAL age;
    REAL dage;
    REAL feh;
    REAL dfeh;
    REAL sigma_v[3];
    REAL pad1;
    REAL rho0;
    REAL hr1;
    REAL hr2;
    REAL xl;
    REAL r_warp;
    REAL r_flare;
    REAL g_warp;
    REAL g_flare;
    REAL th_max;
    REAL qparam;
    REAL bparam;
    REAL ac;
    REAL rhoc;
    REAL eps;
    REAL x0;
    REAL y0;
    REAL z0;
    REAL rc_bulge;
    REAL rot_matrix[3][3];
    REAL agevec[7];
    REAL fehvec[7];
} ClPopParams;

typedef struct {
    REAL pos_c[6];
    REAL cone_dir[3];
    REAL cone_dth;
    REAL app_mag_limits[2];
    REAL color_limits[2];
    REAL abs_mag_limits[2];
    REAL r_max;
    ulong seed;
    uint field_id;
    uint pad;
} ClSurveyParams;

typedef struct {
    REAL pos[3];
    REAL vel[3];
    REAL age;
    REAL feh;
    REAL smass;
    REAL mags[3];
    REAL abs_mags[3];
    REAL rad;
    REAL dmod;
    int popid;
    int accepted;
} ClOutputStar;

// ============================================================================
// Philox4x32 PRNG Implementation
// ============================================================================

inline uint fmix32(uint h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

typedef struct {
    uint key[2];
    uint ctr[4];
    uint buffer[4];
    int buffer_idx;
    REAL normal_cache;
    int has_normal_cache;
} PhiloxState;

inline void philox4x32_round(uint* ctr, const uint* key) {
    ulong prod0 = (ulong)ctr[0] * (ulong)PHILOX_M4X32_0;
    uint hi0 = (uint)(prod0 >> 32);
    uint lo0 = (uint)prod0;

    ulong prod1 = (ulong)ctr[2] * (ulong)PHILOX_M4X32_1;
    uint hi1 = (uint)(prod1 >> 32);
    uint lo1 = (uint)prod1;

    uint c0 = hi1 ^ ctr[1] ^ key[0];
    uint c1 = lo1;
    uint c2 = hi0 ^ ctr[3] ^ key[1];
    uint c3 = lo0;

    ctr[0] = c0;
    ctr[1] = c1;
    ctr[2] = c2;
    ctr[3] = c3;
}

inline void philox4x32_10(const uint* in_ctr, const uint* in_key, uint* out) {
    uint ctr[4];
    ctr[0] = in_ctr[0]; ctr[1] = in_ctr[1]; ctr[2] = in_ctr[2]; ctr[3] = in_ctr[3];
    uint key[2];
    key[0] = in_key[0]; key[1] = in_key[1];

    philox4x32_round(ctr, key);
    for (int i = 1; i < 10; ++i) {
        key[0] += PHILOX_W32_0;
        key[1] += PHILOX_W32_1;
        philox4x32_round(ctr, key);
    }
    out[0] = ctr[0]; out[1] = ctr[1]; out[2] = ctr[2]; out[3] = ctr[3];
}

void philox_init(PhiloxState* s, ulong seed, uint field, uint pop, uint leaf, uint star, uint purpose) {
    s->key[0] = ((uint)seed) ^ fmix32(purpose);
    s->key[1] = ((uint)(seed >> 32)) ^ fmix32(pop);
    s->ctr[0] = 0;
    s->ctr[1] = star;
    s->ctr[2] = leaf;
    s->ctr[3] = field;
    philox4x32_10(s->ctr, s->key, s->buffer);
    s->buffer_idx = 0;
    s->has_normal_cache = 0;
    s->normal_cache = (REAL)0.0;
}

void philox_refill(PhiloxState* s) {
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

inline uint philox_next_u32(PhiloxState* s) {
    if (s->buffer_idx >= 4) {
        philox_refill(s);
    }
    return s->buffer[s->buffer_idx++];
}

inline REAL philox_uniform(PhiloxState* s) {
#if REAL_IS_FP64
    uint w0 = philox_next_u32(s);
    uint w1 = philox_next_u32(s);
    ulong u53 = (((ulong)w0) << 21) | (((ulong)w1) >> 11);
    return ((double)u53 + 0.5) * (1.0 / 9007199254740992.0);
#else
    uint w0 = philox_next_u32(s);
    uint w1 = philox_next_u32(s);
    uint u24 = (w0 >> 8);
    return ((float)u24 + 0.5f) * (1.0f / 16777216.0f);
#endif
}

REAL philox_standard_normal(PhiloxState* s) {
    if (s->has_normal_cache) {
        s->has_normal_cache = 0;
        return s->normal_cache;
    }
    REAL u1 = philox_uniform(s);
    REAL u2 = philox_uniform(s);
    REAL r = sqrt((REAL)(-2.0) * log(u1));
    REAL theta = (REAL)(2.0 * M_PI) * u2;
    REAL z0 = r * cos(theta);
    REAL z1 = r * sin(theta);
    s->normal_cache = z1;
    s->has_normal_cache = 1;
    return z0;
}

// ============================================================================
// Warp, Flare & Density Models
// ============================================================================

void thin_disk_set_pos_b(REAL x, REAL y, REAL z,
                         REAL r_warp, REAL r_flare,
                         REAL g_warp, REAL g_flare, REAL th_max,
                         REAL* pos_b, REAL* k_flare) {
    REAL r = sqrt(x*x + y*y);
    REAL th = atan2(y, x);
    if (th < (REAL)0.0) th += (REAL)(2.0 * M_PI);

    REAL zw = z;
    if (r > r_warp) {
        zw = z - g_warp * (r - r_warp) * cos(th - th_max);
    }

    REAL kf = (r > r_flare) ? (REAL)1.0 + g_flare * (r - r_flare) : (REAL)1.0;

    pos_b[0] = x;
    pos_b[1] = y;
    pos_b[2] = zw;
    *k_flare = kf;
}

inline REAL thin_disk_eps(REAL age) {
    REAL val = (REAL)0.104 * sqrt((age + (REAL)0.1e9) / (REAL)10.1e9);
    return (val < (REAL)0.079) ? val : (REAL)0.079;
}

REAL thin_disk0_density(REAL x, REAL y, REAL z, REAL age, const ClPopParams* pop) {
    REAL pos_b[3];
    REAL k_flare;
    thin_disk_set_pos_b(x, y, z, pop->r_warp, pop->r_flare, pop->g_warp, pop->g_flare, pop->th_max, pos_b, &k_flare);
    REAL eps1 = thin_disk_eps(age);
    REAL denom_z = eps1 * k_flare;
    REAL a2 = pos_b[0]*pos_b[0] + pos_b[1]*pos_b[1] + (pos_b[2]*pos_b[2]) / (denom_z * denom_z);
    return (pop->rho0 / eps1) * (exp(-a2 / pop->hr1) - exp(-a2 / pop->hr2)) / k_flare;
}

REAL thin_disk1_density(REAL x, REAL y, REAL z, REAL age, const ClPopParams* pop) {
    REAL pos_b[3];
    REAL k_flare;
    thin_disk_set_pos_b(x, y, z, pop->r_warp, pop->r_flare, pop->g_warp, pop->g_flare, pop->th_max, pos_b, &k_flare);
    REAL eps1 = thin_disk_eps(age);
    REAL denom_z = eps1 * k_flare;
    REAL a2 = pos_b[0]*pos_b[0] + pos_b[1]*pos_b[1] + (pos_b[2]*pos_b[2]) / (denom_z * denom_z);
    REAL term1 = exp(-sqrt((REAL)0.25 + a2 / pop->hr1));
    REAL term2 = exp(-sqrt((REAL)0.25 + a2 / pop->hr2));
    return (pop->rho0 / eps1) * (term1 - term2) / k_flare;
}

REAL thick_disk_density(REAL x, REAL y, REAL z, const ClPopParams* pop) {
    REAL pos_b[3];
    REAL k_flare;
    thin_disk_set_pos_b(x, y, z, pop->r_warp, pop->r_flare, pop->g_warp, pop->g_flare, pop->th_max, pos_b, &k_flare);
    REAL a2 = sqrt(pos_b[0]*pos_b[0] + pos_b[1]*pos_b[1]);
    REAL hr2 = pop->hr2;
    REAL xl = pop->xl;
    REAL temp = ((REAL)2.0*hr2 + xl)*((REAL)3.0*k_flare*hr2*xl + (REAL)3.0*k_flare*k_flare*hr2*hr2 + xl*xl)
                  / (((REAL)2.0*hr2*k_flare + xl)*((REAL)3.0*hr2*xl + (REAL)3.0*hr2*hr2 + xl*xl));
    REAL radial_part = pop->rho0 * exp(-(a2 - (REAL)8.0) / pop->hr1);
    if (fabs(z) < xl) {
        return radial_part * ((REAL)1.0 - pos_b[2]*pos_b[2] / (xl * ((REAL)2.0*hr2*k_flare + xl))) / temp;
    } else {
        REAL z_scale = hr2 * k_flare;
        REAL factor = exp(xl / z_scale) / ((REAL)1.0 + (REAL)0.5 * xl / z_scale);
        return radial_part * factor * exp(-fabs(pos_b[2]) / z_scale) / temp;
    }
}

REAL spheroid_density(REAL x, REAL y, REAL z, const ClPopParams* pop) {
    REAL a = sqrt(x*x + y*y + (z*z) / (pop->eps * pop->eps));
    if (a < pop->ac) {
        return pop->rhoc;
    } else {
        return pop->rho0 * pow(a / (REAL)8.0, (REAL)(-2.44));
    }
}

REAL bulge_density(REAL x, REAL y, REAL z, const ClPopParams* pop) {
    REAL bx = pop->rot_matrix[0][0]*x + pop->rot_matrix[0][1]*y + pop->rot_matrix[0][2]*z;
    REAL by = pop->rot_matrix[1][0]*x + pop->rot_matrix[1][1]*y + pop->rot_matrix[1][2]*z;
    REAL bz = pop->rot_matrix[2][0]*x + pop->rot_matrix[2][1]*y + pop->rot_matrix[2][2]*z;

    REAL term_xy = bx*bx / (pop->x0 * pop->x0) + by*by / (pop->y0 * pop->y0);
    REAL term_z = bz*bz / (pop->z0 * pop->z0);
    REAL a2 = sqrt(term_xy*term_xy + term_z*term_z);

    REAL cut = sqrt(bx*bx + by*by) - pop->rc_bulge;
    if (cut < (REAL)0.0) {
        return pop->rho0 * exp((REAL)(-0.5) * a2);
    } else {
        return pop->rho0 * exp((REAL)(-0.5) * a2) * exp(-cut*cut / (REAL)0.5);
    }
}

REAL eval_density(int pop_id, REAL x, REAL y, REAL z, REAL age, const ClPopParams* pop) {
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
    return (REAL)0.0;
}

REAL thin_disk_dlnrho_dln_r(int pop_id, REAL x, REAL y, REAL z, REAL age, const ClPopParams* pop) {
    REAL dr = (REAL)1e-3;
    REAL d2 = eval_density(pop_id, x * ((REAL)1.0 + dr), y * ((REAL)1.0 + dr), z, age, pop);
    REAL d1 = eval_density(pop_id, x * ((REAL)1.0 - dr), y * ((REAL)1.0 - dr), z, age, pop);
    return (d2 - d1) / (dr * (d1 + d2));
}

REAL eval_asymmetric_drift(int pop_id, REAL x, REAL y, REAL z, REAL vc, REAL age, REAL sigma_r, const ClPopParams* pop) {
    REAL rc = sqrt(x*x + y*y);
    REAL sig0 = pop->sigma_v[0];
    REAL sig1 = pop->sigma_v[1];
    REAL sig2 = pop->sigma_v[2];

    if (pop_id <= 6) {
        REAL temp = ((REAL)1.0 - sig1*sig1 / (sig0*sig0)) + ((REAL)1.0 - sig2*sig2 / (sig0*sig0));
        temp += thin_disk_dlnrho_dln_r(pop_id, x, y, z, age, pop) - (REAL)2.0 * pop->qparam * rc / (REAL)2.5;
        temp *= sigma_r * sigma_r * (REAL)0.5 / vc;
        return -temp;
    } else if (pop_id == 7) {
        REAL temp = ((REAL)1.0 - sig1*sig1 / (sig0*sig0)) + ((REAL)1.0 - sig2*sig2 / (sig0*sig0));
        temp -= rc * ((REAL)1.0 / pop->hr1 + (REAL)2.0 * pop->qparam / (REAL)2.5);
        temp *= sigma_r * sigma_r * (REAL)0.5 / vc;
        return -temp;
    } else if (pop_id == 8) {
        return vc; // (vc - va) == 0
    } else if (pop_id == 9) {
        return vc - rc * (REAL)71.62;
    }
    return (REAL)0.0;
}

REAL eval_sigmar(int pop_id, REAL age, REAL rc, const ClPopParams* pop) {
    if (pop_id <= 6) {
        REAL val = (REAL)50.0 * pow((age + (REAL)0.1e9) / (REAL)10.1e9, pop->bparam);
        if (val > (REAL)43.1) val = (REAL)43.1;
        return val * exp(((REAL)8.0 - rc) * pop->qparam / (REAL)2.5);
    } else {
        return pop->sigma_v[0];
    }
}

REAL eval_amr(int pop_id, REAL x, REAL y, REAL z, REAL age, const ClPopParams* pop) {
    if (pop_id <= 6) {
        REAL r = sqrt(x*x + y*y);
        REAL feh_interp = pop->fehvec[0];
        if (age <= pop->agevec[0]) {
            feh_interp = pop->fehvec[0];
        } else if (age >= pop->agevec[6]) {
            feh_interp = pop->fehvec[6];
        } else {
            for (int i = 0; i < 6; ++i) {
                if (age >= pop->agevec[i] && age <= pop->agevec[i+1]) {
                    REAL frac = (age - pop->agevec[i]) / (pop->agevec[i+1] - pop->agevec[i]);
                    feh_interp = pop->fehvec[i] + frac * (pop->fehvec[i+1] - pop->fehvec[i]);
                    break;
                }
            }
        }
        return feh_interp - (REAL)0.07 * (r - (REAL)8.0);
    } else {
        return pop->feh;
    }
}

// ============================================================================
// Kinematics Coordinate Transforms
// ============================================================================

void vel_cylindrical_to_cartesian(REAL x, REAL y, REAL v_phi, REAL v_z, REAL v_r,
                                  REAL* vx, REAL* vy, REAL* vz) {
    REAL phi = atan2(y, x);
    REAL sin_phi = sin(phi);
    REAL cos_phi = cos(phi);
    *vx = -v_phi * sin_phi + v_r * cos_phi;
    *vy =  v_phi * cos_phi + v_r * sin_phi;
    *vz =  v_z;
}

void vel_spherical_to_cartesian(REAL x, REAL y, REAL z,
                                REAL v_l, REAL v_b, REAL v_r,
                                REAL* vx, REAL* vy, REAL* vz) {
    REAL r = sqrt(x*x + y*y + z*z);
    REAL l = atan2(y, x);
    if (l < (REAL)0.0) l += (REAL)(2.0 * M_PI);
    REAL b = (r > (REAL)0.0) ? asin(fmax((REAL)(-1.0), fmin((REAL)1.0, z / r))) : (REAL)0.0;

    REAL sin_l = sin(l);
    REAL cos_l = cos(l);
    REAL sin_b = sin(b);
    REAL cos_b = cos(b);

    *vx = -v_l * sin_l - v_b * sin_b * cos_l + v_r * cos_b * cos_l;
    *vy =  v_l * cos_l - v_b * sin_b * sin_l + v_r * cos_b * sin_l;
    *vz =  v_b * cos_b + v_r * sin_b;
}

// ============================================================================
// Tables: IMF and VCirc Interpolation
// ============================================================================

REAL sample_imf(__global const REAL* imf_x, __global const REAL* imf_cpd, uint n, REAL target_cpd) {
    if (n < 2) return (n == 1) ? imf_x[0] : (REAL)0.0;
    if (target_cpd >= imf_cpd[0]) return imf_x[0];
    if (target_cpd <= imf_cpd[n - 1]) return imf_x[n - 1];

    uint lo = 0;
    uint hi = n - 1;
    while (hi - lo > 1) {
        uint mid = (lo + hi) >> 1;
        if (imf_cpd[mid] >= target_cpd) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    REAL c0 = imf_cpd[lo];
    REAL c1 = imf_cpd[lo + 1];
    REAL x0 = imf_x[lo];
    REAL x1 = imf_x[lo + 1];
    REAL dc = c1 - c0;
    if (fabs(dc) < (REAL)1e-30) return x0;
    REAL frac = (target_cpd - c0) / dc;
    return x0 + frac * (x1 - x0);
}

REAL eval_vcirc(__global const REAL* vcirc_r, __global const REAL* vcirc_v, uint n, REAL r) {
    if (n == 0) return (REAL)226.84;
    if (r <= vcirc_r[0]) return vcirc_v[0];
    if (r >= vcirc_r[n - 1]) return vcirc_v[n - 1];

    uint lo = 0;
    uint hi = n - 1;
    while (hi - lo > 1) {
        uint mid = (lo + hi) >> 1;
        if (vcirc_r[mid] <= r) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    REAL r0 = vcirc_r[lo];
    REAL r1 = vcirc_r[lo + 1];
    REAL v0 = vcirc_v[lo];
    REAL v1 = vcirc_v[lo + 1];
    REAL dr = r1 - r0;
    if (fabs(dr) < (REAL)1e-30) return v0;
    return v0 + (v1 - v0) * (r - r0) / dr;
}

int locate_nearest_feh(__global const REAL* feh_grid, uint n, REAL x1) {
    if (n == 0) return 0;
    if (n == 1 || x1 <= feh_grid[0]) return 0;
    if (x1 >= feh_grid[n - 1]) return n - 1;

    uint lo = 0;
    uint hi = n - 1;
    while (hi - lo > 1) {
        uint mid = (lo + hi) >> 1;
        if (feh_grid[mid] <= x1) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    REAL frac = (x1 - feh_grid[lo]) / (feh_grid[lo + 1] - feh_grid[lo]);
    return (frac < (REAL)0.5) ? lo : (lo + 1);
}

// ============================================================================
// Main Star Spawning Kernel
// ============================================================================

__kernel void spawn_kernel(
    ClPopParams pop,
    ClSurveyParams survey,
    __global const ClActiveLeaf* leaves,
    __global const uint* star_to_leaf,
    __global const uint* star_in_leaf,
    __global const REAL* imf_x,
    __global const REAL* imf_cpd,
    uint imf_size,
    __global const REAL* isochrone_m,
    __global const REAL* isochrone_mag0,
    __global const REAL* isochrone_mag1,
    __global const REAL* isochrone_mag2,
    __global const uint* track_offsets,
    __global const REAL* track_m_min,
    __global const REAL* track_m_max,
    uint num_ages,
    uint num_feh,
    REAL age_grid_min,
    REAL d_age,
    __global const REAL* feh_grid,
    __global const REAL* vcirc_r,
    __global const REAL* vcirc_v,
    uint vcirc_size,
    __global ClOutputStar* out_stars,
    uint total_stars
) {
    uint tid = get_global_id(0);
    if (tid >= total_stars) return;

    out_stars[tid].accepted = 0;

    uint leaf_idx = star_to_leaf[tid];
    uint star_idx = star_in_leaf[tid];
    const ClActiveLeaf leaf = leaves[leaf_idx];

    // Seed Philox RNG for this specific star
    PhiloxState rng;
    philox_init(&rng, survey.seed, survey.field_id, pop.pop_id, leaf.orig_leaf_idx, star_idx, 1);

    REAL x = (REAL)0.0, y = (REAL)0.0, z = (REAL)0.0;
    REAL star_age = (REAL)0.0;
    REAL age_linear = (REAL)0.0;

    if (pop.option_e == 1) {
        // Rejection sampling for position inside leaf box
        for (int iter = 0; iter < 100000; ++iter) {
            REAL u1 = philox_uniform(&rng);
            REAL u2 = philox_uniform(&rng);
            REAL u3 = philox_uniform(&rng);
            x = leaf.pos[0] + (u1 - (REAL)0.5) * (REAL)2.0 * leaf.l;
            y = leaf.pos[1] + (u2 - (REAL)0.5) * (REAL)2.0 * leaf.l;
            z = leaf.pos[2] + (u3 - (REAL)0.5) * (REAL)2.0 * leaf.l;
            REAL u4 = philox_uniform(&rng);
            star_age = leaf.pos[3] + (u4 - (REAL)0.5) * (REAL)2.0 * leaf.lage;

            REAL rho = eval_density(pop.pop_id, x, y, z, star_age, &pop);
            REAL u5 = philox_uniform(&rng);
            if (u5 * ((REAL)2.0 * leaf.rho_max) <= rho) {
                break;
            }
        }
        age_linear = star_age;
    } else {
        REAL u_age = philox_uniform(&rng);
        age_linear = pop.age + (u_age - (REAL)0.5) * (REAL)2.0 * pop.dage;
    }

    REAL z0 = philox_standard_normal(&rng);
    REAL feh = eval_amr(pop.pop_id, x, y, z, age_linear, &pop) + z0 * pop.dfeh;

    REAL u_imf = philox_uniform(&rng);
    REAL target_cpd = leaf.cpd_min + u_imf * (leaf.cpd_max - leaf.cpd_min);
    REAL smass = sample_imf(imf_x, imf_cpd, imf_size, target_cpd);
    REAL log10_age = log10(age_linear);

    // Isochrone interpolation
    int i_age = (int)floor((log10_age - age_grid_min) / d_age + (REAL)0.5);
    if (i_age < 0) i_age = 0;
    if (i_age >= (int)num_ages) i_age = num_ages - 1;

    int j_feh = locate_nearest_feh(feh_grid, num_feh, feh);
    uint track_idx = j_feh * num_ages + i_age;

    uint track_start = track_offsets[track_idx];
    uint track_count = track_offsets[track_idx + 1] - track_start;
    REAL m_min = track_m_min[track_idx];
    REAL m_max = track_m_max[track_idx];

    REAL abs_mags[3] = { (REAL)1e6, (REAL)1e6, (REAL)1e6 };
    if (smass > m_min && smass < m_max && track_count >= 2) {
        uint lo = track_start;
        uint hi = track_start + track_count - 1;
        while (hi - lo > 1) {
            uint mid = (lo + hi) >> 1;
            if (isochrone_m[mid] <= smass) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        REAL m0 = isochrone_m[lo];
        REAL m1 = isochrone_m[lo + 1];
        REAL dm = m1 - m0;
        REAL frac = (fabs(dm) > (REAL)1e-30) ? (smass - m0) / dm : (REAL)0.0;
        abs_mags[0] = isochrone_mag0[lo] + frac * (isochrone_mag0[lo + 1] - isochrone_mag0[lo]);
        abs_mags[1] = isochrone_mag1[lo] + frac * (isochrone_mag1[lo + 1] - isochrone_mag1[lo]);
        abs_mags[2] = isochrone_mag2[lo] + frac * (isochrone_mag2[lo + 1] - isochrone_mag2[lo]);
    }

    if (abs_mags[0] >= (REAL)1e4) {
        return; // Unphysical or below track limit
    }

    if (pop.option_e == 0) {
        // Rejection sampling for position occurs here for option_e == 0
        for (int iter = 0; iter < 100000; ++iter) {
            REAL u1 = philox_uniform(&rng);
            REAL u2 = philox_uniform(&rng);
            REAL u3 = philox_uniform(&rng);
            x = leaf.pos[0] + (u1 - (REAL)0.5) * (REAL)2.0 * leaf.l;
            y = leaf.pos[1] + (u2 - (REAL)0.5) * (REAL)2.0 * leaf.l;
            z = leaf.pos[2] + (u3 - (REAL)0.5) * (REAL)2.0 * leaf.l;

            REAL rho = eval_density(pop.pop_id, x, y, z, pop.age, &pop);
            REAL u5 = philox_uniform(&rng);
            if (u5 * ((REAL)2.0 * leaf.rho_max) <= rho) {
                break;
            }
        }
    }

    // Kinematics
    REAL age_temp = pow((REAL)10.0, log10_age);
    REAL rc = sqrt(x*x + y*y);
    REAL vc = eval_vcirc(vcirc_r, vcirc_v, vcirc_size, rc);
    REAL temp = eval_sigmar(pop.pop_id, age_temp, rc, &pop);

    REAL z1 = philox_standard_normal(&rng);
    REAL z2 = philox_standard_normal(&rng);
    REAL z3 = philox_standard_normal(&rng);

    REAL sig_u = pop.sigma_v[0];
    REAL sig_v = pop.sigma_v[1];
    REAL sig_w = pop.sigma_v[2];

    REAL va = eval_asymmetric_drift(pop.pop_id, x, y, z, vc, age_temp, temp, &pop);

    REAL v3 = z1 * temp * sig_v / sig_u - (vc - va);
    REAL v4 = z2 * temp * sig_w / sig_u;
    REAL v5 = z3 * temp;

    REAL vx, vy, vz;
    if (pop.pop_id == 8) {
        vel_spherical_to_cartesian(x, y, z, v3, v4, v5, &vx, &vy, &vz);
    } else {
        vel_cylindrical_to_cartesian(x, y, v3, v4, v5, &vx, &vy, &vz);
    }

    // Shift to Heliocentric frame
    REAL hx = x - survey.pos_c[0];
    REAL hy = y - survey.pos_c[1];
    REAL hz = z - survey.pos_c[2];

    REAL hvx = vx - survey.pos_c[3];
    REAL hvy = vy - survey.pos_c[4];
    REAL hvz = vz - survey.pos_c[5];

    REAL rad = sqrt(hx*hx + hy*hy + hz*hz);
    REAL dmod = (REAL)5.0 * log10(rad * (REAL)100.0);

    REAL mags[3];
    mags[0] = abs_mags[0] + dmod;
    mags[1] = abs_mags[1] + dmod;
    mags[2] = abs_mags[2] + dmod;

    // Survey cuts
    if (mags[0] < survey.app_mag_limits[0] || mags[0] > survey.app_mag_limits[1]) return;
    if ((mags[1] - mags[2]) < survey.color_limits[0] || (mags[1] - mags[2]) > survey.color_limits[1]) return;
    if ((mags[0] - dmod) < survey.abs_mag_limits[0] || (mags[0] - dmod) > survey.abs_mag_limits[1]) return;
    if (rad > survey.r_max) return;

    // Observer patch cone check
    if (rad > (REAL)0.0) {
        REAL dot = (hx * survey.cone_dir[0] + hy * survey.cone_dir[1] + hz * survey.cone_dir[2]) / rad;
        if (dot > (REAL)1.0) dot = (REAL)1.0;
        if (dot < (REAL)(-1.0)) dot = (REAL)(-1.0);
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
