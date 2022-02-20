import numpy as np
import math
import matplotlib.pyplot as plt
import copy

pts = [
[0.4540,0.7126],
[0.4828,0.7299],
[0.4828,0.4368],
[0.5115,0.4368],
[0.5172,0.4253],
[0.5287,0.7471],
[0.5460,0.7644],
[0.5517,0.4195],
[0.5862,0.7931],
[0.5862,0.4138],
[0.5977,0.3908],
[0.6092,0.6839],
[0.6149,0.6552],
[0.6149,0.3793],
[0.6322,0.8161],
[0.6207,0.7241],
[0.6322,0.5747],
[0.6322,0.5517],
[0.6322,0.5230],
[0.6322,0.3563],
[0.6437,0.8391],
[0.6552,0.7586],
[0.6552,0.6609],
[0.6494,0.5862],
[0.6437,0.5057],
[0.6494,0.4828],
[0.6552,0.3448],
[0.6667,0.8506],
[0.6667,0.4540],
[0.6667,0.4253],
[0.6667,0.3218],
[0.6782,0.8736],
[0.6897,0.7931],
[0.6897,0.6897],
[0.6782,0.5747],
[0.6782,0.4138],
[0.6782,0.3793],
[0.6897,0.3448],
[0.6897,0.3103],
[0.6954,0.8908],
[0.7011,0.5517],
[0.7011,0.8161],
[0.7011,0.7126],
[0.7356,0.8966],
[0.7241,0.8621],
[0.7126,0.8391],
[0.7241,0.7241],
[0.7126,0.5402],
[0.7241,0.5230],
[0.7126,0.2989],
[0.7184,0.2874],
[0.7299,0.9310],
[0.7586,0.9655],
[0.7471,0.9368],
[0.7471,0.9080],
[0.7586,0.7529],
[0.7529,0.4828],
[0.7701,0.9885],
[0.7816,1.0000],
[0.7874,0.7701],
[0.7874,0.5862],
[0.7816,0.5517],
[0.7816,0.5172],
[0.7931,0.4828],
[0.7931,0.4540],
[0.7931,0.6897],
[0.7931,0.6552],
[0.7931,0.6207],
[0.8046,0.7126],
[0.8161,0.7299],
[0.8161,0.4713],
[0.8218,0.4425],
[0.8276,0.4195],
[0.8333,0.7816],
[0.8391,0.7471],
[0.8621,0.7874],
[0.8506,0.7701],
[0.8851,0.7931]]

#Scalling
def scalar (s_x, s_y, matrix_2d):
    scallar_matrix = [[s_x, 0],[0, s_y]]
    return np.matmul(scallar_matrix, matrix_2d)

#Rotation
def rotate (theta, matrix_2d):
    rotate_matrix = [[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]]
    return np.matmul(rotate_matrix, matrix_2d)

#Shear
def shear (alpha, matrix_2d):
    shear_matrix = [[1, alpha],[0, 1]]
    return np.matmul(shear_matrix, matrix_2d)

s_x = 2
s_y = 2
theta = math.pi/2
shear_value = 5

scalar_result = []
rotate_result = []
shear_result = []

for i in range (len(pts)):
    x = copy.copy(pts[i][0])
    y = copy.copy(pts[i][1])
    scalar_result.append(scalar(s_x, s_y, [x, y]))
    x = copy.copy(pts[i][0])
    y = copy.copy(pts[i][1])
    rotate_result.append(rotate(theta, [x, y]))
    x = copy.copy(pts[i][0])
    y = copy.copy(pts[i][1])
    shear_result.append(shear(shear_value, [x, y]))

for point in pts:
    plt.plot(pts[0], pts[1], 'go--', linewidth=2, markersize=2)
plt.show()

for point in scalar_result:
    plt.plot(point[0], point[1], 'go--', linewidth=2, markersize=2)
plt.show()


for point in rotate_result:
    plt.plot(point[0], point[1], 'go--', linewidth=2, markersize=2)
plt.show()

for point in shear_result:
    plt.plot(point[0], point[1], 'go--', linewidth=2, markersize=2)
plt.show()

def combineTogether():
    scalar_result = []
    rotate_result = []
    final_result = []
    for i in range (len(pts)):
        x = pts[i][0]
        y = pts[i][1]
        scalar_result.append(scalar(s_x, s_y, [x, y]))
    for i in range (len(scalar_result)):
        x = scalar_result[i][0]
        y = scalar_result[i][1]
        rotate_result.append(rotate(theta, [x, y]))
    for i in range (len(rotate_result)):
        x = rotate_result[i][0]
        y = rotate_result[i][1]
        final_result.append(shear(shear_value, [x, y]))
    return final_result

final_result = combineTogether()
for point in final_result:
    plt.plot(point[0], point[1], 'go--', linewidth=2, markersize=2)
plt.show()