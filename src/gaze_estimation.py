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
            (Vector3D): 3D Vector (x, y z)
        '''
        # Extract the coordinates of the 3D vector
        x = output[0,0]
        y = output[0,1]
        z = output[0,2]

        return Vector3D(x, y, z)

