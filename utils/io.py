from plyfile import PlyData
import numpy as np

def read_ply(ply_file):
    '''
    读取一个ply文件中的mesh
    '''
    read_data = PlyData.read(ply_file)
    verts_x = np.array(read_data['vertex']['x']).reshape(-1, 1)
    verts_y = np.array(read_data['vertex']['y']).reshape(-1, 1)
    verts_z = np.array(read_data['vertex']['z']).reshape(-1, 1)
    verts = np.concatenate([verts_x, verts_y, verts_z], axis=-1)
    
    face = np.stack(read_data['face']['vertex_indices'])
    
    return verts, face

