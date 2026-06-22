import numpy as np
from plyfile import PlyData

input_ply_path = "input.ply"  
output_bin_path = "scene_data.bin"

max_vertices_to_extract = 100000  

print("Loading PLY file into memory...")
plydata = PlyData.read(input_ply_path)
vertex = plydata['vertex']
total_vertices = len(vertex)
print(f"Total vertices in PLY file: {total_vertices}")

# Determine how many points to actually process
num_vertices = min(total_vertices, max_vertices_to_extract)
print(f"Extracting first {num_vertices} vertices...")

# Slice the data to extract only the requested amount
vertex_slice = vertex[:num_vertices]

print("Processing positions...")
x, y, z = vertex_slice['x'], vertex_slice['y'], vertex_slice['z']

print("Converting Spherical Harmonics (f_dc) to RGB color...")
SH_C0 = 0.28209479177387814
r = np.clip(0.5 + SH_C0 * vertex_slice['f_dc_0'], 0.0, 1.0)
g = np.clip(0.5 + SH_C0 * vertex_slice['f_dc_1'], 0.0, 1.0)
b = np.clip(0.5 + SH_C0 * vertex_slice['f_dc_2'], 0.0, 1.0)

print("Converting log-scale to linear scale...")
scale_0 = np.exp(vertex_slice['scale_0'])
scale_1 = np.exp(vertex_slice['scale_1'])
scale_2 = np.exp(vertex_slice['scale_2'])

print("Converting logit-opacity to sigmoid activation...")
opacity = 1.0 / (1.0 + np.exp(-vertex_slice['opacity']))

print("Processing rotation (WXYZ -> XYZW)...")
rot_w, rot_x, rot_y, rot_z = vertex_slice['rot_0'], vertex_slice['rot_1'], vertex_slice['rot_2'], vertex_slice['rot_3']

print("Packing data into binary matrix...")
packed_data = np.zeros((num_vertices, 14), dtype=np.float32)

packed_data[:, 0:3]   = np.stack([x, y, z], axis=-1)
packed_data[:, 3:6]   = np.stack([scale_0, scale_1, scale_2], axis=-1)
packed_data[:, 6:10]  = np.stack([rot_x, rot_y, rot_z, rot_w], axis=-1)
packed_data[:, 10]    = opacity
packed_data[:, 11:14] = np.stack([r, g, b], axis=-1)

print("Writing to disk...")
packed_data.tofile(output_bin_path)

print(f"Done! Extracted {num_vertices} vertices and saved to: {output_bin_path}")
print(f"Final file size: {packed_data.nbytes / (1024*1024):.2f} MB")
