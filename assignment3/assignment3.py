from turtle import width
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple, Optional
import numpy as np
import imageio
from scipy.spatial import Delaunay

#For pts data
FILE_NAME_POINTS_1 = "1a.pts"
FILE_NAME_POINTS_2 = "100b.pts"
FILE_NAME_FACE_1 = "1a.jpg"
FILE_NAME_FACE_2 = "100b.jpg"

#For asf data
FILE_NAME_POINTS_1_ASF = "05-5m.asf"
FILE_NAME_POINTS_2_ASF = "08-1f.asf"
FILE_NAME_FACE_1_ASF = "05-5m.jpg"
FILE_NAME_FACE_2_ASF = "08-1f.jpg"

#Common
FILE_PATH_POINTS = "./data/points/"
FILE_PATH_FACE = "./data/face/"
FILE_OUTPUT = "./data/output/"
MAX_ALPHA_VALUE = 300 #Actual maximum is 255

class Vertex:
    def __init__(self,
                 number : int,
                 x : float,
                 y : float):
        self.number = number
        self.x = x
        self.y = y

    def get_points_in_list(self):
        return [self.x, self.y]

    def get_points_in_array(self):
        return np.array([self.x, self.y])

    def get_number(self):
        return self.number

class Triangle:
    def __init__(self, vertices: List[Vertex], vertices_number : List[int], triangle_num):
        self.triangle_num = triangle_num
        self.vertices_number = vertices_number
        self.vertices = vertices

    def get_verticses(self):
        return self.vertices

    def get_vertex_numbers(self):
        return self.vertices_number

    def get_vertex_points(self):
        point_lists = []
        for vertex in self.vertices:
            vertex : Vertex
            point_lists.append(vertex.get_points_in_list())
        return np.array(point_lists)

class Point_Sets:
    def __init__(self):
        self.vertices : List[Vertex] = []
        self.number_of_sets = 0
        self.image_file_name = ""

    def get_vertices(self) -> List[Vertex]:
        return self.vertices

    def add_vertex(self, vertex : Vertex):
        self.vertices.append(vertex)

    def set_length(self, length):
        self.number_of_sets = length

    def set_image_name(self, fileName):
        self.image_file_name = fileName
    
    def length(self):
        return self.number_of_sets

    def get_image_name(self):
        return self.image_file_name

    def convert_to_array(self):
        point_lists = []
        for vertex in self.vertices:
            vertex : Vertex
            point_lists.append(vertex.get_points_in_list())
        return np.array(point_lists)

class Graphics:
    def __init__(self, x_limit, y_limit):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Average faces')
        self.x_limit = x_limit
        self.y_limit = y_limit
        plt.xlim( -0.1, x_limit + 0.1)
        plt.ylim( -0.1, y_limit + 0.1)
    
    def set_background(self, filePath, fileName_1, fileName_2, alpha_value):
        img_1 = Image.open(filePath + fileName_1)
        img_2 = Image.open(filePath + fileName_2)
        img_1.putalpha(int(alpha_value * MAX_ALPHA_VALUE))
        img_2.putalpha(int((1 - alpha_value) * MAX_ALPHA_VALUE))
        self.ax.imshow(img_1, extent=[0, self.x_limit, 0, self.y_limit], 
          cmap='gray',
          alpha = alpha_value)
        self.ax.imshow(img_2, extent=[0, self.x_limit, 0, self.y_limit],
          cmap='gray',
          alpha = 1 - alpha_value)

    def draw_points(self, point_list : np.ndarray, color : str):
        self.ax.scatter(point_list[:,0], point_list[:,1], marker='o', s=1 ,c=color)
    
    #Notice that additional line needs to be drawn
    def draw_edges_with_triangles(self, triangles: List[Triangle]):
        for triangle in triangles:
            triangle : Triangle
            points = triangle.get_vertex_points()
            plt.plot(points[:,0], points[ :,1], color='green', linewidth=0.4)

    def draw_edges_with_given(self, points: np.ndarray):
        plt.plot(points[:,0], points[:,1], color='green', linewidth=0.4)

    def draw_edges_with_delaunay(self, points: np.ndarray):
        tri = Delaunay(points)
        plt.triplot(points[:,0], points[:,1], tri.simplices, color='green', linewidth=0.4)
        return tri.simplices

    #Takes single pyplot data and export to an image file
    def export_image(self, file_index):
        file_name = FILE_OUTPUT + "output" + str(file_index)
        self.fig.savefig(file_name)
        print("Exported")
        self.close_figure()
        return file_name + ".png"

    def close_figure(self):
        plt.close()

class FileHandler:
    def __init__(self):
        self.output_file_names = []
        self.file_index = 0

    #For asf media type coordination files (x,y range from 0 to 1).
    #Note that sets are given inverse of y of normal pyplot range
    def read_asf(self, filePath , filename) -> Point_Sets:
        new_lines = []
        sets = Point_Sets()
        with open(filePath + filename) as f:
            lines = f.readlines()
        for line in lines:
            if line == None:
                continue
            if len(line) <= 1:
                continue
            line = line.strip()
            if (line[0] != '#'):
                new_lines.append(line)
        data_size = int(new_lines[0])
        for i in range(data_size):
            key_words = (new_lines[i + 1].replace("\t", "")).split(" ")
            sets.add_vertex(Vertex(int(key_words[4]), np.float16(key_words[2]), 1 - np.float16(key_words[3])))
        
        #Add corners
        sets.add_vertex(Vertex((data_size + 1), np.float16(.0), np.float16(.0)))
        sets.add_vertex(Vertex((data_size + 2), np.float16(1.0), np.float16(.0)))
        sets.add_vertex(Vertex((data_size + 3), np.float16(.0), np.float16(1.0)))
        sets.add_vertex(Vertex((data_size + 4), np.float16(1.0), np.float16(1.0)))
        sets.add_vertex(Vertex((data_size + 5), np.float16(.5), np.float16(.0)))
        sets.add_vertex(Vertex((data_size + 6), np.float16(1.0), np.float16(.5)))
        sets.add_vertex(Vertex((data_size + 7), np.float16(.5), np.float16(1.0)))
        sets.add_vertex(Vertex((data_size + 8), np.float16(.0), np.float16(.5)))
        sets.set_length(data_size + 8)
        sets.set_image_name(new_lines[len(new_lines) - 1])
        return sets

    def set_output_file_names(self, list):
        self.output_file_names = list

    #Reads images based on the exported sequence and make those into gif file
    def make_gif_file(self):
        images = []
        for filename in self.output_file_names:
            images.append(imageio.imread(filename))
        imageio.mimsave( FILE_OUTPUT + 'movie.gif', images, fps=10)

class Calculation:
    #Alpha value given between 0 and 1
    def average_shape(point_set_1, point_set_2, n):
        average_point_set = []
        afine_set =[]
        for i in range(n):
            alpha = i / n
            afine_set.append(alpha)
            average_point_set.append(np.add(((1-alpha) * point_set_1), alpha * point_set_2))
        return average_point_set, afine_set

    #Inputs two lines and returns single interpretted line
    def calculate_average_edge_vector(alpha, line_1 : Vertex, line_2 : Vertex):
        points_1 = line_1.get_points_in_array()
        points_2 = line_2.get_points_in_array()
        return points_1 + alpha * (points_2 - points_1)

    #Inputs two triangles and returns single interpretted triangle
    def calculate_average_triangle_vectors(alpha, start_triangles : List[Triangle], end_triangles : List[Triangle]) \
            -> List[Triangle]:
        start_triangles : Triangle
        end_triangles : Triangle
        calculated_triangles : List[Triangle] = []
        new_triangle_number = 0
        zipped_triangles = zip(start_triangles, end_triangles)
        
        for start_triangle, end_triangle in zipped_triangles:
            start__triangle_vertices: List[Vertex] = start_triangle.get_verticses() #3 Vertices
            end__triangle_vertices : List[Vertex] = end_triangle.get_verticses() #3 Vertices
            zipped_vertices = zip(start__triangle_vertices, end__triangle_vertices)

            calculated_triangle_vertices : List[Vertex] = []
            new_list_of_indices = []
            for start_vertex, end_vertex in zipped_vertices: #Calculate 3 edges of single triangle
                start_point = start_vertex.get_points_in_array()
                end_point = end_vertex.get_points_in_array()
                e_vector = lambda a: start_point + a * (end_point - start_point) #Must contain x, y coordinates\
                calculated_triangle_vertices.append(Vertex(start_vertex.get_number(),
                                                    np.float16(e_vector(alpha)[0]),
                                                    np.float16(e_vector(alpha)[1])))
                new_list_of_indices.append(start_vertex.get_number())
            new_triangle = Triangle(calculated_triangle_vertices , new_list_of_indices , new_triangle_number)
            new_triangle_number += 0
            calculated_triangles.append(new_triangle)
        return calculated_triangles #Return shifted single-frame triangles


file_handler = FileHandler()

#File read
face_coordination_list_1 : Point_Sets = file_handler.read_asf(FILE_PATH_POINTS, FILE_NAME_POINTS_1_ASF)
face_coordination_list_2 : Point_Sets = file_handler.read_asf(FILE_PATH_POINTS, FILE_NAME_POINTS_2_ASF)

#File export intialization
file_index = 0
file_name = []

#Point read
vertices_ndarray_1 = face_coordination_list_1.convert_to_array()
vertices_1 : List[Vertex] = face_coordination_list_1.get_vertices()

vertices_ndarray_2 = face_coordination_list_2.convert_to_array()
vertices_2 : List[Vertex] = face_coordination_list_2.get_vertices()

'''
    Initial Draw
'''
#640, 480
graphics = Graphics(1, 1)
graphics.set_background(FILE_PATH_FACE,
                        FILE_NAME_FACE_1_ASF,
                        FILE_NAME_FACE_2_ASF,
                        1)

#Draw initial face triangles with spicy library and retrieve points
start_triangles_points = graphics.draw_edges_with_delaunay(vertices_ndarray_1)
#Draw points
graphics.draw_points(vertices_ndarray_1, "red")
graphics.draw_points(vertices_ndarray_2, "blue")

#Result of a single frame
file_name.append(graphics.export_image(file_index))
file_index += 1

#Triangles
triangles_set_start : List[Triangle] = []
triangles_set_end : List[Triangle] = []

#Get identical triangular points for target face from start face
triangle_count = 0
for triangle_points in start_triangles_points:
    triangle_count += 1
    triangle_vertices_start = [vertices_1[triangle_points[0]], vertices_1[triangle_points[1]], vertices_1[triangle_points[2]]]
    triangle_vertices_end = [vertices_2[triangle_points[0]], vertices_2[triangle_points[1]], vertices_2[triangle_points[2]]]
    triangle_indices_start = [triangle_points[0], triangle_points[1], triangle_points[2]]
    new_triangle_start = Triangle(triangle_vertices_start, triangle_indices_start, triangle_count)
    new_triangle_end = Triangle(triangle_vertices_end, triangle_indices_start, triangle_count)
    triangles_set_start.append(new_triangle_start)
    triangles_set_end.append(new_triangle_end)

#average sets for points, affine for gradual increase in value from 0 to 1 by 1/n
average_sets, affine_set = Calculation.average_shape(vertices_ndarray_1 ,vertices_ndarray_2, face_coordination_list_1.length())

whole_frames_triangles : List[List[Triangle]] = []

for affine in affine_set:
    single_frame_triangles : List[Triangle] = Calculation.calculate_average_triangle_vectors(affine, triangles_set_start, triangles_set_end)
    whole_frames_triangles.append(single_frame_triangles)

zipped_object = zip(affine_set, whole_frames_triangles)

#At this point, triangle frame and affine set has no actual interactions
for affine, single_frame_triangles in zipped_object:
    single_frame_triangles : List[Triangle]
    #640, 480
    graphics = Graphics(1, 1)
    graphics.set_background(FILE_PATH_FACE,
                            FILE_NAME_FACE_1_ASF,
                            FILE_NAME_FACE_2_ASF,
                            1 - affine)
    graphics.draw_edges_with_triangles(single_frame_triangles)
    graphics.draw_points(vertices_ndarray_1, "red")
    graphics.draw_points(vertices_ndarray_2, "blue")

    #Result of a single frame
    file_name.append(graphics.export_image(file_index))
    file_index += 1

#Create gif fil
file_handler.set_output_file_names(file_name)
file_handler.make_gif_file()

print("done")