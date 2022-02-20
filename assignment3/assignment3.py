from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio  

FILE_PATH_POINTS = "./data/points/"
FILE_NAME_POINTS_1 = "1a.pts"
FILE_NAME_POINTS_2 = "100b.pts"
FILE_PATH_FACE = "./data/face/"
FILE_NAME_FACE_1 = "1a.jpg"
FILE_NAME_FACE_2 = "100b.jpg"
FILE_OUTPUT = "./data/output/"
MAX_ALPHA_VALUE = 255 

class Graphics:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Average faces')
        plt.xlim(0, 250)
        plt.ylim(0, 300)
    
    def set_background(self, filePath, fileName_1, fileName_2, alpha_value):
        img_1 = Image.open(filePath + fileName_1)
        img_2 = Image.open(filePath + fileName_2)
        img_1.putalpha(int(alpha_value * MAX_ALPHA_VALUE))
        img_2.putalpha(int((1 - alpha_value) * MAX_ALPHA_VALUE))
        self.ax.imshow(img_1, extent=[0, 250, 0, 300], cmap='gray', alpha = alpha_value)
        self.ax.imshow(img_2, extent=[0, 250, 0, 300], cmap='gray', alpha = 1 - alpha_value)

    def draw_points(self, point_list, color):
        return self.ax.scatter(point_list[:,0], point_list[:,1], s=(1.0 * 1.0) ,c=color)

    #Takes single pyplot data and export to an image file
    def export_image(self, file_index):
        file_name = FILE_OUTPUT + "output" + str(file_index)
        self.fig.savefig(file_name)
        print("Exported")
        self.close_figure()
        return file_name + ".png"

    def close_figure(self):
        plt.close()

class ReadData:
    def read_pts(filePath , filename):
        data = np.loadtxt((filePath + filename),
            comments = ("version:", "n_points:", "{", "}"))
        return data

class FileHandler:
    def __init__(self, output_file_names):
        self.output_file_names = output_file_names
        self.file_index = 0

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
        for i in range(n):
            alpha = i / n
            average_point_set.append(np.add(((1-alpha) * point_set_1), alpha * point_set_2))
        return average_point_set
        
    def triangulation():
        pass

#Read two face coordination sets
face_coordination_list_1 = ReadData.read_pts(FILE_PATH_POINTS, FILE_NAME_POINTS_1)
face_coordination_list_2 = ReadData.read_pts(FILE_PATH_POINTS, FILE_NAME_POINTS_2)

#Get sets of average points between two images
average_point_set = Calculation.average_shape(face_coordination_list_1,
                                              face_coordination_list_2,
                                              len(face_coordination_list_1))

#Draw scatter plots on the image
index = 0
graphics = Graphics()
file_name = []
graphics.set_background(FILE_PATH_FACE, FILE_NAME_FACE_1, FILE_NAME_FACE_2, 1)
graphics.draw_points(face_coordination_list_1, "red")
graphics.draw_points(face_coordination_list_2, "blue")
file_name.append(graphics.export_image(index))

index += 1

for average_point in average_point_set:
    alpha = index / len(face_coordination_list_1)
    graphics = Graphics()
    
    #Keed the original and target face info
    graphics.set_background(FILE_PATH_FACE, FILE_NAME_FACE_1, FILE_NAME_FACE_2, 1- alpha)
    graphics.draw_points(face_coordination_list_1, "red") 
    graphics.draw_points(face_coordination_list_2, "blue")
    graphics.draw_points(average_point, "green")
    file_name.append(graphics.export_image(index))
    index += 1

graphics = Graphics()
graphics.set_background(FILE_PATH_FACE, FILE_NAME_FACE_1, FILE_NAME_FACE_2, 0)
graphics.draw_points(face_coordination_list_1, "red")
graphics.draw_points(face_coordination_list_2, "blue")
file_name.append(graphics.export_image(index))

#Create gif fil
file_handler = FileHandler(file_name)
file_handler.make_gif_file()

print("done")