import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple, Optional
import numpy as np
import imageio
from scipy.spatial import Delaunay
from PIL import Image

#For asf data: x, y coodinates have numeric value between 0 to 1.
FILE_NAME_POINTS_1_ASF = "04-5m.asf"
FILE_NAME_POINTS_2_ASF = "09-1m.asf"
FILE_NAME_FACE_1_ASF = "04-5m.jpg"
FILE_NAME_FACE_2_ASF = "09-1m.jpg"

#Common
FILE_PATH_POINTS = "./data/points/"
FILE_PATH_FACE = "./data/face/"
FILE_OUTPUT = "./data/output/"

def image_import(file):
    image = Image.open(file, 'r')
    width, height = image.size            # width, height in pixel
    pix_data_list = list(image.getdata()) # single list of rgb data
    #reform data into 2d array
    pix_coord_data_list = []
    index = 0
    for y in range(height):
        row = []
        for x in range(width):
            row.append(pix_data_list[index])
            index += 1
        pix_coord_data_list.append(row)
    
    return width, height, pix_coord_data_list, image

departure_rgb_width, departure_rgb_height, departure_rgb_data, departure_img\
    = image_import(FILE_PATH_FACE + FILE_NAME_FACE_1_ASF)

arrival_rgb_width, arrival_rgb_height, arrival_rgb_data, arrival_img\
    = image_import(FILE_PATH_FACE + FILE_NAME_FACE_2_ASF)

print(len(departure_rgb_data[0]), len(departure_rgb_data))
print(len(arrival_rgb_data[0]), len(arrival_rgb_data))

'''
x, y: coordination points.
number: unique id for edge location from other vertex.
''' 
class Vertex():
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

'''  
vertices: list of Vertex - must have lenght of max 3.
    ex) [Vertex(x1, y1), Vertex(x2, y2), Vertex(x3, y3)]
vertices_number :list of corelated edge vertex number - assigned from Delaunay
    ex) [2, 4, 3]
triangle_number : number of unique triangle id
'''
class Triangle():
    def __init__(self, vertices: List[Vertex], vertices_number : List[int], triangle_num):
        self.triangle_num = triangle_num
        self.vertices_number = vertices_number
        self.vertices = vertices
        self.matrix = self.toMatrix()
        
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
    '''
        if triangle points are given like (x1, y1)
        returns
        [[x1, y1, 0],
         [x2, y2, 0],
         [x3, y3, 1]]
         
        [[x1, x2, x3],
         [y1, y2, y3],
         [0, 0, 1]]
    '''
    def toMatrix(self):
        return [[self.vertices[0].x, self.vertices[0].y, 0],
                [self.vertices[1].x, self.vertices[1].y, 0],
                [self.vertices[2].x, self.vertices[2].y, 1]]

    def toSymMatrix(self):
        return [[self.vertices[0].x, self.vertices[1].x, self.vertices[2].x],
                [self.vertices[0].y, self.vertices[1].y, self.vertices[2].y],
                [0, 0, 1]]
    
    def getMatrix(self):
        return self.matrix
    
    '''
        if triangle points are given like (x1, y1)
        returns
        [[x1 y1]
         [x2 y2]
         [x3 y3]]
         
         used for simplices
    '''
    def getMatrixArray(self):
        return np.array([element[:2] for element in self.matrix])
    
    def area(self, x1, y1, x2, y2, x3, y3):
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                    + x3 * (y1 - y2)) / 2.0)
 
    def isInside(self, x, y):
        triangle_points = self.getMatrixArray()
        x1 = triangle_points[0][0]
        y1 = triangle_points[0][1]
        x2 = triangle_points[1][0]
        y2 = triangle_points[1][1]
        x3 = triangle_points[2][0]
        y3 = triangle_points[2][1]
        
        A = self.area (x1, y1, x2, y2, x3, y3)
        A1 = self.area (x, y, x2, y2, x3, y3)
        A2 = self.area (x1, y1, x, y, x3, y3)
        A3 = self.area (x1, y1, x2, y2, x, y)

        # is same as A
        if(A == A1 + A2 + A3):
            return True
        else:
            return False

class Point_Sets():
    def __init__(self):
        self.vertices : List[Vertex] = []
        self.number_of_sets = 0
        self.image_file_name = ""

    def get_vertices(self) -> List[Vertex]:
        return self.vertices

    def add_vertex(self, vertex : Vertex):
        self.vertices.append(vertex)
            
    def convert_to_array(self):
        point_lists = []
        for vertex in self.vertices:
            vertex : Vertex
            point_lists.append(vertex.get_points_in_list())
        return np.array(point_lists)
    
    def set_length(self, length):
        self.number_of_sets = length
        
    def get_length(self):
        return self.number_of_sets

    def set_image_name(self, fileName):
        self.image_file_name = fileName
        
    def print_points(self):
        for i in range(self.number_of_sets):
            tempVertex : Vertex = self.vertices[i]
            if (tempVertex == None):
                print("Sth wrong")
            print(str(tempVertex.x) + " : " + str(tempVertex.y))

'''
File handler can perform:
    - Read asf files
    - Delaunary points propagtions in below format.
        Format: <path#> <type> <x rel.> <y rel.> <point#> <connects from> <connects to> 
        with the number of points leaded ahead.
    - Creates gif file based on given output file names.
'''
class FileHandler():
    def __init__(self):
        self.output_file_names = []
        self.file_index = 0

    def read_asf(self, filePath : str , filename : str, x_multiplier, y_multiplier) -> Point_Sets:
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
            #Only takes vertex id, x, y points -> Until below 4 decimal points: float16
            sets.add_vertex(Vertex(int(key_words[4]), np.float16(key_words[2]) * x_multiplier,
                                                      np.float16(np.float16(key_words[3])) * y_multiplier))
        
        #Add corners(Manual coordination)
        sets.add_vertex(Vertex((data_size + 1), np.float16(.0)  , np.float16(.0)))
        sets.add_vertex(Vertex((data_size + 2), np.float16(x_multiplier), np.float16(.0)))
        sets.add_vertex(Vertex((data_size + 3), np.float16(.0) , np.float16(y_multiplier)))
        sets.add_vertex(Vertex((data_size + 4), np.float16(x_multiplier) , np.float16(y_multiplier)))
        sets.add_vertex(Vertex((data_size + 5), np.float16(x_multiplier/2), np.float16(.0)))
        sets.add_vertex(Vertex((data_size + 6), np.float16(x_multiplier), np.float16(y_multiplier/2)))
        sets.add_vertex(Vertex((data_size + 7), np.float16(x_multiplier/2), np.float16(y_multiplier)))
        sets.add_vertex(Vertex((data_size + 8), np.float16(.0), np.float16(y_multiplier/2)))
        
        #increase size
        sets.set_length(data_size + 8)
        
        #??? Why save image name with length?
        sets.set_image_name(new_lines[len(new_lines) - 1])
        return sets

    #Add image file for gif file generation
    def add_output_file_names(self, name):
        self.output_file_names = name
    
    #Set image files for gif file generation
    def set_output_file_names(self, name_list):
        self.output_file_names = name_list

    #Reads images based on the exported sequence and make those into gif file
    def make_gif_file(self):
        images = []
        for filename in self.output_file_names:
            images.append(imageio.imread(filename))
        imageio.mimsave( FILE_OUTPUT + 'movie.gif', images, fps=10)

file_handler = FileHandler()

#Points_Sets instances
face_coordination_list_1 : Point_Sets = file_handler.read_asf(FILE_PATH_POINTS, FILE_NAME_POINTS_1_ASF,\
                                                              departure_rgb_width, departure_rgb_height)
face_coordination_list_2 : Point_Sets = file_handler.read_asf(FILE_PATH_POINTS, FILE_NAME_POINTS_2_ASF,\
                                                              arrival_rgb_width, arrival_rgb_height)
#Points in array
vertices_ndarray_1 = face_coordination_list_1.convert_to_array()
vertices_ndarray_2 = face_coordination_list_2.convert_to_array()

class Graphics:
    def __init__(self, x_limit, y_limit):
        self.fig = plt.figure()
        self.x_limit = x_limit
        self.y_limit = y_limit
        plt.title('Morph faces')
        plt.xlim( -0.1, x_limit + 0.1)
        plt.ylim( -0.1, y_limit + 0.1)
        plt.gca().invert_yaxis() #This resolves image flipping problem!!
        
    
    #Setting two images as a signle frame with given opacity value.
    def set_background(self, filePath, fileName_1, alpha_value):
        img_1 = Image.open(filePath + fileName_1)
        img_1 = img_1.transpose(method=Image.FLIP_TOP_BOTTOM)
        plt.imshow(img_1, extent=[0, self.x_limit, 0, self.y_limit], 
          cmap='gray',
          alpha = alpha_value)

    def draw_points(self, point_list : np.ndarray, color : str):
        plt.scatter(point_list[:,0], point_list[:,1], marker='o', s=1 ,c=color)
    
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
    
    def draw_edges_with_single_triangle(self, triangle: Triangle, line_width: float):
        points : np.ndarray = triangle.getMatrixArray()
        tri = Delaunay(points)
        plt.triplot(points[:,0], points[:,1], tri.simplices, color='green', linewidth=line_width)
    
    def show_img(self,image):
        plt.imshow(image)
    
    #Takes single pyplot data and export to an image file
    def export_image(self, file_index):
        file_name = FILE_OUTPUT + "output" + str(file_index)
        self.fig.savefig(file_name)
        self.close_figure()
        return file_name + ".png"

    def read_color(self, file):
        image = Image.open(file, 'r')
        width, height = image.size            # width, height in pixel
        pix_data_list = list(image.getdata()) # single list of rgb data
        return width, height, pix_data_list, image
    
    def close_figure(self):
        plt.close()
        
    def show(self):
        plt.show()

graphics_1 = Graphics(departure_rgb_width, departure_rgb_height)

#Set background
graphics_1.set_background(FILE_PATH_FACE, FILE_NAME_FACE_1_ASF, 1)
graphics_1.draw_points(vertices_ndarray_1, "red")

#Crucial data to form triangles
departure_face_triangles_points = graphics_1.draw_edges_with_delaunay(vertices_ndarray_1)

graphics_2 = Graphics(departure_rgb_width, departure_rgb_height)

#Set background
graphics_2.set_background(FILE_PATH_FACE, FILE_NAME_FACE_2_ASF, 1)
graphics_2.draw_points(vertices_ndarray_2, "blue")

#Crucial data to form triangles
arrival_face_triangles_points = graphics_2.draw_edges_with_delaunay(vertices_ndarray_2)

departure_points_list = face_coordination_list_1.get_vertices()
arrival_points_list = face_coordination_list_2.get_vertices()

#List of Triangle objects
departure_triangle_list = []
arrival_triangle_list = []

#Triangle coordination list
#iterate every triangle points
index = 0 #For triangle number
for tri_points in departure_face_triangles_points: #Triangle points from Delaunary
    departure_temp_vetices = []
    arrival_temp_vetices = []
    for point in tri_points:
        departure_temp_vetices.append(departure_points_list[point])
        arrival_temp_vetices.append(arrival_points_list[point])
    #Create Triangle object and append to triangle list
    departure_triangle_list.append(Triangle(departure_temp_vetices, tri_points.tolist() ,index))
    arrival_triangle_list.append(Triangle(arrival_temp_vetices, tri_points.tolist() ,index))

#Make sure they have the same number of triangles
print("Amount of triangle objects", len(departure_triangle_list))
print("Amount of triangle objects", len(arrival_triangle_list))

image_data = [[0] * arrival_rgb_width for i in range(arrival_rgb_height)]
test_graphics = Graphics(arrival_rgb_width, arrival_rgb_height)


affine_rate = 0.5

#Returns triangle object
def get_triangle_C_by_rate(triangle_A : Triangle, triangle_B : Triangle, affine_rate):
    triangle_A_array = triangle_A.get_vertex_points()
    triangle_B_array = triangle_B.get_vertex_points()
    
    
    #Vectors shpae A -> shape B
    V_a = np.array([triangle_A_array[0][0] - triangle_B_array[0][0], triangle_A_array[0][1] - triangle_B_array[0][1]])
    V_b = np.array([triangle_A_array[1][0] - triangle_B_array[1][0], triangle_A_array[1][1] - triangle_B_array[1][1]])
    V_c = np.array([triangle_A_array[2][0] - triangle_B_array[2][0], triangle_A_array[2][1] - triangle_B_array[2][1]])

    #Somewhere in the middle(affine_rate) Shape A -> Shape B
    C_a = triangle_B_array[0] + V_a * affine_rate
    C_b = triangle_B_array[1] + V_b * affine_rate
    C_c = triangle_B_array[2] + V_c * affine_rate
    
    #No need for unique id : -1
    vertex_a = Vertex(-1, C_a[0], C_a[1])
    vertex_b = Vertex(-1, C_b[0], C_b[1])
    vertex_c = Vertex(-1, C_c[0], C_c[1])
    
    return Triangle([vertex_a, vertex_b, vertex_c], triangle_A.get_vertex_numbers(), triangle_A.triangle_num)

triangle_C = get_triangle_C_by_rate(departure_triangle_list[0], arrival_triangle_list[0], affine_rate)
print(triangle_C.toMatrix())


inverse_C = np.linalg.inv(np.array(triangle_C.toMatrix()))
inverse_A = np.matmul(inverse_C, departure_triangle_list[0].toMatrix())
inverse_B = np.matmul(inverse_C, arrival_triangle_list[0].toMatrix())
print("Inverse A: \n",inverse_A)
print("Inverse B: \n",inverse_B)

affine_rate = 0.5
morph_traiangles = []
for i in range(len(departure_triangle_list)):
    morph_traiangles.append(get_triangle_C_by_rate(departure_triangle_list[i], arrival_triangle_list[i], affine_rate))


def draw_entire(triangles):

    for triangle in triangles:
        for y in range(arrival_rgb_height - 1):
            for x in range(arrival_rgb_width - 1):
                if triangle.isInside(x, y) == True:
                    #Get RGB data from A and B

                    A_rgb_coord = np.matmul(np.array(inverse_A), [np.float16(x), np.float16(y), 1])[:2] #A color coordination
                    B_rgb_coord = np.matmul(np.array(inverse_B), [np.float16(x), np.float16(y), 1])[:2] #B color coordination
                    print(A_rgb_coord)
                    print(B_rgb_coord)
                    #RGB values -> x and y location are flipped: Couldn't know why
                    #but somehow the coordinates are mapped in symmatrical way
                        
                    #A_rgb = np.array(departure_rgb_data[int(A_rgb_coord[0])][int(A_rgb_coord[1])])
                    #B_rgb = np.array(arrival_rgb_data[int(B_rgb_coord[0])][int(B_rgb_coord[1])])
                    #c_rgb = (A_rgb * affine_rate + B_rgb * (1 - affine_rate))
                    #image_data[y][x] = (np.array([int(c_rgb[0]), int(c_rgb[1]), int(c_rgb[2])]))

draw_entire(morph_traiangles)

    
test_graphics.show_img(image_data)