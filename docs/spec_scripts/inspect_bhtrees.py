# SPDX-License-Identifier: AGPL-3.0-or-later
import glob
import os
import struct
GALAXIA_DATA_DIR = os.environ.get("GALAXIA_DATA_DIR", "GalaxiaData")

def inspect_bhtrees():
    tree_dir = os.path.join(GALAXIA_DATA_DIR, "BHTree-2.2/bhtree_with_wf")
    files = sorted(glob.glob(os.path.join(tree_dir, "bhtree_*_E*.ebf")))
    
    print(f"{'File':<30} {'PopID':<6} {'OptionE':<8} {'Nodes':<10} {'Leafs':<10} {'FileSize(MB)':<12}")
    print("-" * 80)
    for fpath in files:
        fname = os.path.basename(fpath)
        # Parse popID and optionE
        # e.g. bhtree_0_E1.ebf -> pop 0, E1
        parts = fname.replace(".ebf", "").split("_")
        pop_id = parts[1]
        opt_e = parts[2]
        
        # Read EBF items to get /Nodes and /Leafs dims
        nodes_dim = 0
        leafs_dim = 0
        with open(fpath, "rb") as f:
            file_size = f.seek(0, 2)
            f.seek(0)
            while f.tell() < file_size:
                pos = f.tell()
                sig = f.read(8)
                if len(sig) < 8:
                    break
                ver = f.read(4)
                unused4 = struct.unpack("<8i", f.read(32))
                flags = f.read(4)
                capacity = struct.unpack("<q", f.read(8))[0]
                endian_test, headersize, namesize, datatype, datasize, rank, unitsize, sdefsize = unused4
                if rank > 0:
                    dims = struct.unpack(f"<{rank}q", f.read(8 * rank))
                else:
                    dims = (1,)
                name = f.read(namesize).decode('utf-8', errors='replace').lower()
                
                curr = f.tell()
                extra = (pos + headersize) - curr
                if extra > 0:
                    f.seek(extra, 1)
                datapos = f.tell()
                
                if name == "/nodes":
                    nodes_dim = dims[0]
                elif name == "/leafs":
                    leafs_dim = dims[0]
                    
                f.seek(datapos + capacity)
                
        fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"{fname:<30} {pop_id:<6} {opt_e:<8} {nodes_dim:<10} {leafs_dim:<10} {fsize_mb:<12.2f}")

if __name__ == "__main__":
    inspect_bhtrees()
