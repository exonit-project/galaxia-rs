# SPDX-License-Identifier: AGPL-3.0-or-later
import struct
import numpy as np
import os
GALAXIA_DATA_DIR = os.environ.get("GALAXIA_DATA_DIR", "GalaxiaData")

def read_ebf_item(filepath, target_key):
    target_key = target_key.lower()
    fsize = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        while f.tell() < fsize:
            pos = f.tell()
            sig = f.read(6)
            if len(sig) < 6:
                break
            f.seek(pos)
            
            if sig.startswith(b"EBF"):
                # Ver 1.0
                sig6 = f.read(6)
                name = f.read(100).split(b'\x00')[0].split(b' ')[0].decode('ascii', errors='replace').lower()
                ver2 = f.read(2)
                f.read(36)
                endian_test, datatype, datasize, rank = struct.unpack("<4i", f.read(16))
                f.read(32)
                dims = struct.unpack(f"<{rank}q", f.read(8 * rank))
                f.seek(pos + 256)
                datapos = f.tell()
                elems = 1
                for d in dims:
                    elems *= d
                capacity = elems * datasize
            else:
                sig8 = f.read(8)
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
                unit = f.read(unitsize).decode('utf-8', errors='replace')
                sdef = f.read(sdefsize).decode('utf-8', errors='replace')
                curr = f.tell()
                extra = (pos + headersize) - curr
                if extra > 0:
                    f.seek(extra, 1)
                datapos = f.tell()
                elems = 1
                for d in dims:
                    elems *= d
                    
            if name == target_key:
                # read data
                dtype_map = {
                    1: np.dtype('S1'),
                    2: np.dtype('<i4'),
                    3: np.dtype('<i8'),
                    4: np.dtype('<f4'),
                    5: np.dtype('<f8'),
                }
                dt = dtype_map.get(datatype, np.dtype('<f4'))
                arr = np.frombuffer(f.read(elems * datasize), dtype=dt).reshape(dims)
                return arr
                
            f.seek(datapos + capacity)
    raise KeyError(f"Key {target_key} not found in {filepath}")

if __name__ == "__main__":
    for path, key in [
        (os.path.join(GALAXIA_DATA_DIR, 'Extinction/ExMap3d_1024.ebf'), '/exmap3d.xmms'),
        (os.path.join(GALAXIA_DATA_DIR, 'Extinction/ExMap3d_1024.ebf'), '/exmap2d.xmms'),
        (os.path.join(GALAXIA_DATA_DIR, 'Extinction/Schlegel_4096.ebf'), '/exmap2d.xmms'),
        (os.path.join(GALAXIA_DATA_DIR, 'Model/surface_density.ebf'), '/surface_density1.xmms'),
        (os.path.join(GALAXIA_DATA_DIR, 'Model/surface_density.ebf'), '/surface_density2.xmms'),
    ]:
        print(f"=== {path} : {key} ===")
        val = read_ebf_item(path, key)
        print(val)
