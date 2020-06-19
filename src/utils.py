from collections import namedtuple

BoundingBox = namedtuple("BoundingBox", "l c x y w h")
# where l is the label, c the confidence score, 
# (x,y) are the coordinates of the top-left corner of the bounding box 
# and w its width and h its height in pixels.

PoseAngles = namedtuple("PoseAngles", "y p r")
# where y is the yaw, p the pitch and r the roll