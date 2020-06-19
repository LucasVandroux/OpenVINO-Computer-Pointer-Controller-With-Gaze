from collections import namedtuple
from math import cos, sin

import cv2
import numpy as np

BoundingBox = namedtuple("BoundingBox", "l c x y w h")
# where l is the label, c the confidence score, 
# (x,y) are the coordinates of the top-left corner of the bounding box 
# and w its width and h its height in pixels.

PoseAngles = namedtuple("PoseAngles", "y p r")
# where y is the yaw, p the pitch and r the roll

Landmark = namedtuple("Landmark", "n x y")
# where n is the name of the landmark,
# (x,y) are the coordinates of the landmark

def draw_3Daxis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    '''
    Function to draw the 3D axis to represent a pose.
    Source: https://github.com/natanielruiz/deep-head-pose/blob/master/code/utils.py

    Args:
        img (numpy.array): image on which to draw the 3D axis
        yaw (float): yaw angle of the pose
        pitch (float): pitch angle of the pose
        roll (float): roll angle of the pose
        tdx (int: None): x coordinate for the origin of the 3D axis
        tdy (int: None): y coordinate for the origin of the 3D axis
        size (int: 100): length of the axis

    Returns:
        img (numpy.array): returns input image with the 3D axis
    '''
    # Convert the angles to radians
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # Set the origin of the 3D axis
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis pointing down. drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    # Draw the axis
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img