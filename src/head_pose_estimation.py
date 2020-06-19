from model import OpenVINOModel
from utils import PoseAngles, draw_3Daxis

'''
Documentation: https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html

Download: python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001
'''

class HeadPoseEstimationModel(OpenVINOModel):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def predict(self, image):
        '''
        Estimate the Pose Angles of a Head.
        
        Args:
        image (numpy.array BGR): image of the head to estimate the pose angles of.

        Returns:
        results (PoseAngles): yaw, pitch and roll of the head.
        '''
        # Pre process the image
        input_image = self.preprocess_input(image)

        # Estimate the Head Pose
        output = self.inference(input_image)
        
        # Postprocess the output of the model
        results = self.postprocess_output(output)

        return results

    def postprocess_output(self, output):
        '''
        Postprocss the output of the model.

        Args:
            output (dict): dict containing the output.

        Returns:
            (PoseAngles): yaw, pitch and roll of the head.
        '''
        # Extract the yaw in degree [-90,90]
        yaw = output['angle_y_fc'][0,0]

        # Extract the pitch in degree [-70,70]
        pitch = output['angle_p_fc'][0,0]

        # Extract the roll in degree [-70,70]
        roll = output['angle_r_fc'][0,0]

        return PoseAngles(yaw, pitch, roll)

    def display_output(self, image, results):
        '''
        Display the 3D axis to represent the head pose.

        Args:
            image (numpy.array): original image used for the inference.
            results (PoseAngles): yaw, pitch and roll of the head.

        Returns:
            image_out (numpy.array): copy of the input image with the 3D axis
                representing the head pose.
        '''
        image_out = image.copy()
        
        image_out = draw_3Daxis(
            image_out,
            results.y,
            results.p,
            results.r
        )

        return image_out

