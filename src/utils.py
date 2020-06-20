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

Vector3D = namedtuple("Vector3D", "x y z")
# where (x, y, z) are the coordinates of the Vector

def draw_3Daxis(image, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    '''
    Function to draw the 3D axis to represent a pose.
    Source: https://github.com/natanielruiz/deep-head-pose/blob/master/code/utils.py

    Args:
        image (numpy.array): image on which to draw the 3D axis
        yaw (float): yaw angle of the pose
        pitch (float): pitch angle of the pose
        roll (float): roll angle of the pose
        tdx (int: None): x coordinate for the origin of the 3D axis
        tdy (int: None): y coordinate for the origin of the 3D axis
        size (int: 100): length of the axis

    Returns:
        image (numpy.array): returns input image with the 3D axis
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
        height, width = image.shape[:2]
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
    cv2.line(image, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(image, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(image, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return image

def extract_landmark_roi(name, landmarks, roi_size, image, origin_x = 0, origin_y = 0):
    '''
    Extract the ROI around a landmark

    Args:
        name (str): name of the landmark to extract.
        landmarks (list[Landmark]): list of Landmark.
        roi_size (int): size of the ROI [roi_size x roi_size].
        image (numpy.array): image to extract the ROI from.
        origin_x (int: 0): to move the origin if the landmark was 
            extracted from a different ROI.
        origin_y (int: 0): to move the origin if the landmark was 
            extracted from a different ROI.
    
    Returns:
        landmark_roi (numpy.array): extracted ROI around the landmark
        landmark_bbox (BoundingBox): bounding box containing all the 
            information of the ROI

    '''
    # Extract the landmark 
    landmark = [l for l in landmarks if l.n == name][0]

    # Find the coordinates of the landmark in the frame
    x = origin_x + landmark.x
    y = origin_y + landmark.y

    # Calculate the roi_halfsize
    roi_halfsize = int(roi_size / 2)

    # Find the x limits
    x_min = x - roi_halfsize
    x_max = x + (roi_size - roi_halfsize)

    # Find the y limits
    y_min = y - roi_halfsize
    y_max = y + (roi_size - roi_halfsize)

    # Extract the ROI from the frame
    landmark_roi = image[y_min:y_max, x_min:x_max, :]

    # Create a Bounding Box out the ROI
    landmark_bbox = BoundingBox(name, 1, x_min, y_min, int(x_max-x_min), int(y_max-y_min))

    return landmark_roi, landmark_bbox
