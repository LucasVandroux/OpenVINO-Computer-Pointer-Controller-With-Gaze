from model import OpenVINOModel
from utils import PoseAngles
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
        Estimate the Pose Angles of a Head
        
        Args:
        image (numpy.array BGR): image of the head to estimate the pose angles of.

        Returns:
        results (PoseAngles): yaw, pitch and roll of the head
        '''
        output = self.inference(image)

        results = self.postprocess_output(output)

        return results

    def postprocess_output(self, output):
        '''
        Postprocss the output of the model

        Args:
            output (dict): dict containing the output

        Returns:
            (PoseAngles): yaw, pitch and roll of the head
        '''
        yaw = output['angle_y_fc'][0,0]
        pitch = output['angle_p_fc'][0,0]
        roll = output['angle_r_fc'][0,0]

        return PoseAngles(yaw, pitch, roll)

