from math import sqrt

import cv2
import numpy as np

from model import OpenVINOModel
from utils import Vector3D, draw_3Daxis

'''
Documentation: https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html

Download: python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002
'''

class GazeEstimationModel(OpenVINOModel):
    '''
    Class for the Gaze Estimation Model.
    '''
    def inference(self, input_left_eye_image, input_right_eye_image, head_pose_angles):
        '''
        Do the inference using one image of each eye and the pose of the head.
        Overwrite of the parent function.
        
        Args:
            input_left_eye_image (numpy.array): image of the left eye already preprocessed.
            input_right_eye_image (numpy.array): image of the right eye already preprocessed.
            head_pose_angles (numpy.array): array [1x3] head pose [yaw, pitch, and roll]

        Returns:
            output : return the output of the model 
        '''
        # Create the input dictionary
        input_dict = {
            'left_eye_image': input_left_eye_image,
            'right_eye_image': input_right_eye_image,
            'head_pose_angles': head_pose_angles}

        # Do the inference
        output = self.exec_network.infer(input_dict)

        return output
    
    def get_input_shape(self):
        '''
        Gets the input shape of the network.
        Hardcoded due to the multiple numbers of inputs.
        '''
        return [1, 3 , 60, 60]

    def predict(self, left_eye_image, right_eye_image, head_angles):
        '''
        Estimate the Gaze Direction of a Huamn Head.
        
        Args:
            left_eye_image (numpy.array BGR): image of the left eye.
            right_eye_image (numpy.array BGR): image of the right eye.
            head_angles (PoseAngles): yaw, pitch and roll of the head.

        Returns:
            results (Vector3D): 3D Vector (x, y z)
        '''
        # Pre process the image of the eyes
        input_left_eye_image = self.preprocess_input(left_eye_image)
        input_right_eye_image = self.preprocess_input(right_eye_image)

        # Prepare the head_pose_angles
        head_pose_angles = np.array([head_angles.y, head_angles.p, head_angles.r])
        
        # Estimate the Gaze
        output = self.inference(input_left_eye_image, input_right_eye_image, head_pose_angles)
        
        # Postprocess the output of the model
        results = self.postprocess_output(output['gaze_vector'])

        return results

    def postprocess_output(self, output):
        '''
        Postprocss the output of the model.

        Args:
            output (numpy.array): array [1, 3] containing the coordinates
                of the 3D vector representing the gaze direction

        Returns:
            (Vector3D): 3D Vector (x, y z) not normalized
        '''
        # Extract the coordinates of the 3D vector
        x = output[0,0]
        y = output[0,1]
        z = output[0,2]

        return Vector3D(x, y, z)
    
    def display_output(self, image, norm_gaze_vector, list_bbox_eye, color = (255, 0, 0), size = 5000):
        """
         Display the gaze vector from both eye on the image.

        Args:
            image (numpy.array): original image used for the inference.
            norm_gaze_vector(numpy.array): size [3] normalized vector of the gaze direction
            list_bbox_eye (list[BoundingBox]): list of the eyes bounding boxes
            color ((B, G, R): Blue): color to draw the vector in
            size (int: 5000): length of the gaze vector to display

        Returns:
            image_out (numpy.array): copy of the input image with the gaze vector displayed
        """
        # Copy the input image
        image_out = image.copy()

        # Calaculate distance from start to end according to size
        distance = sqrt(size)
        
        # Draw the vector on each eye
        for bbox_eye in list_bbox_eye:
            # Extract the start vector (eye position)
            x = bbox_eye.x + int(bbox_eye.w / 2)
            y = bbox_eye.y + int(bbox_eye.h / 2)
            start_vector = np.array([x, y, 0])

            # Find the gaze vector
            gaze_vector = start_vector + norm_gaze_vector * distance

            # draw the vector
            cv2.arrowedLine(
                image_out,
                (x, y),
                (int(gaze_vector[0]), int(gaze_vector[1])),
                color,
                thickness = 2,
                tipLength = 0.5,
            )
        
        return image_out


