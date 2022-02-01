
import math
from operator import matmul
from re import M
from matplotlib import cm
import plotly.graph_objects as go
from plyfile import PlyData, PlyElement
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

plydata = PlyData.read('./assignment2/data/cube.ply')

v = np.array(plydata.elements[0].data)

#print(plydata)
f = plydata['face'].data['vertex_indices']

def get_x_y_z_vertics(v):
    x_list = []
    y_list = []
    z_list = []
    for coord in v:
        x_list.append(coord[0])
        y_list.append(coord[1])
        z_list.append(coord[2])
    return x_list, y_list, z_list

#Change to only have triangular coordinations
#If polygon points are ginven more than 3 coordinations(traingular),
# change that coordination to triangular coordination
def make_3d_face_coord(faces):
    triangular_face = []
    for face in faces:
        main_point = 0
        if (len(face) > 3):
            index = 1 #Fixed point
            while(index <= len(face) - 2):
                new_list = [face[main_point], face[index], face[index + 1]]
                index += 1
                triangular_face.append(np.array(new_list))
        else:
            triangular_face.append(face)
    return triangular_face

def get_ijk_face_coordinate(faces):
    i_face = []
    j_face = []
    k_face = [] 
    for face in faces:
        i_face.append(face[0])
        j_face.append(face[1])
        k_face.append(face[2])
    return i_face, j_face, k_face

def use_3D_model(x, y, z, x_list, y_list, z_list, i_face , j_face, k_face):
    return go.Figure(data=[go.Mesh3d(
            # 8 vertices of a cube
            x=x_list,
            y=y_list,
            z=z_list,
            colorscale=[[0, 'gold'],
                        [0.5, 'mediumturquoise'],
                        [1, 'magenta']],
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity = np.linspace(0, 1, 8, endpoint=False),
            
            # i, j and k give the vertices of triangles
            i = i_face,
            j = j_face,
            k = k_face,        
            name='y',
            showscale=False)])

#Transformation
def resize_3d(matrix, resize_factor):
    resize_matrix_factor = np.identity(3) * resize_factor
    return np.matmul(resize_matrix_factor, matrix)
'''
#Rotation - x, y, z
def rotation_3d(matrix, yaw_degree, pitch_degree, roll_degree):
    yaw_matrix = np.array([[math.cos(yaw_degree), -math.sin(yaw_degree), 0],
                           [math.sin(yaw_degree), math.cos(yaw_degree), 0],
                           [0, 0, 1]])

    pitch_matrix = np.array([[math.cos(pitch_degree), 0, math.sin(pitch_degree)],
                             [0, 1, 0],
                             [-math.sin(yaw_degree), 0, math.cos(pitch_degree)]])

    roll_matrix = np.array([[1, 0, 0],
                            [0, math.cos(roll_degree), -math.sin(roll_degree)],
                            [0, math.sin(roll_degree), math.cos(roll_degree)]])
    rotational_factor_matrix = np.matmul(np.matmul(yaw_matrix,
                                                   pitch_matrix),
                                                   roll_matrix)
    return np.matmul(matrix, rotational_factor_matrix)

'''
'''
#3d to 2d project - removes z-axis
#n by 3 to n by 2
def third_dimension_to_two_dimension(matrix):
    # 3 X 2 identity
    matrix_reduction_factor = [[1, 0],
                               [0, 1],
                               [0, 0]]
                        
    return np.matmul(matrix, matrix_reduction_factor)
'''

def single_frame(frame_rate):
    resized_v = []

    for coord in v:
        resized_v.append(resize_3d(coord.tolist(), frame_rate + 1))

    x, y, z = get_x_y_z_vertics(v)

    X, Y, Z = get_x_y_z_vertics(resized_v)

    #face coordination
    new_face = make_3d_face_coord(f)
    i_face, j_face, k_face = get_ijk_face_coordinate(new_face)

    #Use this for 3d view - free view mode
    return use_3D_model(x, y, z, X, Y, Z, i_face, j_face, k_face)
    
figure = single_frame(1)
figure.show()
