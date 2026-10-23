# SPDX-License-Identifier: AGPL-3.0-or-later
import glob
import os
import sys
GALAXIA_DATA_DIR = os.environ.get("GALAXIA_DATA_DIR", "GalaxiaData")

def inspect_isochrones():
    padova_dir = os.path.join(GALAXIA_DATA_DIR, "Isochrones/padova/DCMC")
    # Find all output_*.dat files
    dat_files = sorted(glob.glob(os.path.join(padova_dir, "output_*.dat")))
    print(f"Number of metallicity files (FeH): {len(dat_files)}")
    
    # Read one file to get ages and points per isochrone
    f0 = dat_files[0]
    ages = set()
    total_lines = 0
    points_per_age = {}
    with open(f0, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 16:
                total_lines += 1
                age_val = float(parts[0])
                ages.add(age_val)
                points_per_age[age_val] = points_per_age.get(age_val, 0) + 1
                
    num_ages = len(ages)
    num_feh = len(dat_files)
    total_isochrones = num_ages * num_feh
    print(f"Number of age bins: {num_ages} (min={min(ages)}, max={max(ages)})")
    print(f"Total isochrone tracks (Age x FeH): {total_isochrones}")
    avg_points_raw = total_lines / num_ages
    print(f"Raw points per isochrone: average ~{avg_points_raw:.1f}")
    
    # In Galaxia, dwarfs are added (15 points per isochrone if dwarfOn=1)
    num_dwarfs = 15
    avg_points_with_dwarfs = avg_points_raw + num_dwarfs
    print(f"Points per isochrone with dwarfs: ~{avg_points_with_dwarfs:.1f}")
    
    # Total points across all isochrones
    total_points = total_isochrones * avg_points_with_dwarfs
    print(f"Total points in grid: ~{total_points:,.0f}")
    
    # In memory:
    # In spawn (extraFieldsOn=0):
    # m, Mag0, Mag1, Mag2, Teff, Grav, Lum, Mact, Vmon -> 9 doubles per point = 72 bytes/point
    # Plus vector overhead, etc.
    mem_spawn_bytes = total_points * 9 * 8
    print(f"Estimated IsochroneDB grid memory (spawn, extraFieldsOn=0): ~{mem_spawn_bytes / (1024*1024):.1f} MB")
    
    # In append1 (extraFieldsOn=1):
    # adds Mags[4] (I, J, H, Ks) -> 4 more doubles per point = 13 doubles/point = 104 bytes/point
    mem_append1_bytes = total_points * 13 * 8
    print(f"Estimated IsochroneDB grid memory (append1, extraFieldsOn=1): ~{mem_append1_bytes / (1024*1024):.1f} MB")

    # Also inspect extinction maps
    ext_3d = os.path.join(GALAXIA_DATA_DIR, "Extinction/ExMap3d_1024.ebf")
    ext_2d = os.path.join(GALAXIA_DATA_DIR, "Extinction/Schlegel_4096.ebf")
    print("\n=== Extinction Maps ===")
    from inspect_data import inspect_ebf
    inspect_ebf(ext_3d)
    inspect_ebf(ext_2d)

if __name__ == "__main__":
    inspect_isochrones()
