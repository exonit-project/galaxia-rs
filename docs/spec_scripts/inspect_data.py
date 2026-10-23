# SPDX-License-Identifier: AGPL-3.0-or-later
import struct
import os
GALAXIA_DATA_DIR = os.environ.get("GALAXIA_DATA_DIR", "GalaxiaData")

def inspect_ebf(filepath):
    print(f"\n=== Inspecting {filepath} ===")
    fsize = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        items = []
        while f.tell() < fsize:
            pos = f.tell()
            sig = f.read(6)
            if len(sig) < 6:
                break
            f.seek(pos)
            
            # Check if Ver 1.0 or Ver 1.1
            if sig.startswith(b"EBF"):
                # Ver 1.0
                sig6 = f.read(6)
                name_raw = f.read(100).split(b' ')[0].decode('ascii', errors='replace').lower()
                ver2 = f.read(2)
                f.read(36)
                endian_test, datatype, datasize, rank = struct.unpack("<4i", f.read(16))
                f.read(32)
                dims = struct.unpack(f"<{rank}q", f.read(8 * rank))
                f.seek(pos + 256)
                datapos = f.tell()
                
                # elements
                elems = 1
                for d in dims:
                    elems *= d
                capacity = elems * datasize
                unit = ""
                name = name_raw
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
                
            items.append({
                "name": name,
                "unit": unit,
                "datatype": datatype,
                "datasize": datasize,
                "rank": rank,
                "dims": dims,
                "capacity": capacity,
                "datapos": datapos
            })
            f.seek(datapos + capacity)
            
    for it in items:
        print(f"Key: {it['name']:<28} DtypeCode: {it['datatype']} Datasize: {it['datasize']} Dims: {it['dims']} Unit: '{it['unit']}'")
    return items

if __name__ == "__main__":
    inspect_ebf(os.path.join(GALAXIA_DATA_DIR, "Extinction/ExMap3d_1024.ebf"))
    inspect_ebf(os.path.join(GALAXIA_DATA_DIR, "Extinction/Schlegel_4096.ebf"))
    inspect_ebf(os.path.join(GALAXIA_DATA_DIR, "Model/surface_density.ebf"))
