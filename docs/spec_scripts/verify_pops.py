# SPDX-License-Identifier: AGPL-3.0-or-later
# Script to verify and document all constants of the 10 populations

populations = [
    {"id": 0, "name": "ThinDisk0", "class": "ThinDisk0", "optionE": 1},
    {"id": 1, "name": "ThinDisk1", "class": "ThinDisk1", "optionE": 1},
    {"id": 2, "name": "ThinDisk2", "class": "ThinDisk1", "optionE": 1},
    {"id": 3, "name": "ThinDisk3", "class": "ThinDisk1", "optionE": 1},
    {"id": 4, "name": "ThinDisk4", "class": "ThinDisk1", "optionE": 1},
    {"id": 5, "name": "ThinDisk5", "class": "ThinDisk1", "optionE": 1},
    {"id": 6, "name": "ThinDisk6", "class": "ThinDisk1", "optionE": 1},
    {"id": 7, "name": "ThickDisk", "class": "ThickDisk", "optionE": 0},
    {"id": 8, "name": "Spheroid", "class": "Spheroid", "optionE": 0},
    {"id": 9, "name": "Bulge", "class": "Bulge", "optionE": 0},
]

print("Populations:")
for p in populations:
    print(p)
