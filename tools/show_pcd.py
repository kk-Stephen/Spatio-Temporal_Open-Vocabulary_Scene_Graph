import open3d as o3d
m = o3d.io.read_triangle_mesh(r"H:\DATA\replicate\Replica\data\apartment_0\apartment_0\mesh.ply")
m.compute_vertex_normals()
o3d.visualization.draw_geometries([m])