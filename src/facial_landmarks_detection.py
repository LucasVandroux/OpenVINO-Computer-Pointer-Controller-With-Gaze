from model import OpenVINOModel
from utils import Landmark

import cv2
'''
Documentation: https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html

Download: python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009
'''


class FacialLandmarksDetectionModel(OpenVINOModel):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def predict(self, image):
        '''
        Find the position of the two eyes, the nose, the two corner of the mouth 
        
        Args:
        image (numpy.array BGR): image of the head where to find the landmarks

        Returns:
        results (list[Landmark]): list of the landmarks (n, x , y)
        '''
        # Pre process the image
        input_image = self.preprocess_input(image)

        # Extract the landmarks position 
        output = self.inference(input_image)
        
        # Postprocess the output of the model
        results = self.postprocess_output(output[self.output_blob], image)

        return results

    def postprocess_output(self, output, image):
        '''
        Postprocss the output of the model

        Args:
            output (numpy.array): array [1, 10, 1, 1] containing the landmarks' position
            image (numpy.array): original image used for the inference

        Returns:
            (PoseAngles): yaw, pitch and roll of the head
        '''
        # Extract the dimension on the input image
        image_h, image_w, _ = image.shape
        
        # List of the different landmarks in the order outputed by the model
        list_landmarks_name = [
            "left_eye", "right_eye", "nose", "left_mouth", "right_mouth"
        ]

        # Initialize the list to store the detected landmarks
        list_landmarks = []

        # Extract the position of the landmarks
        for idx in range(len(list_landmarks_name)):
            # Name of the landmark
            name = list_landmarks_name[idx]

            # Extract position of the landmark and unormalized them
            x = int(output[0,idx*2,0,0] * image_w)
            y = int(output[0,idx*2+1,0,0] * image_h)

            # Add the landmark to the list of detected landmarks
            list_landmarks.append(Landmark(name, x, y))

        return list_landmarks

    def display_output(self, image, results, color = (0, 0, 255), display_name = False):
        '''
        Display the bounding boxes on the image.

        Args:
            image (numpy.array): original image used for the inference.
            results (list[Landmark]): list of the facial landmarks
            color ((B, G, R): Red): color to draw the bounding boxes
            display_name (Bool: False): display the name of the landmarks

        Returns:
            image_out (numpy.array): copy of the input image with the bounding 
                boxes of the detected head
        '''
        # Copy the input image
        image_out = image.copy()

        for landmark in results:
            # Mark the landmark on the face
            cv2.circle(
                image_out, 
                (landmark.x, landmark.y), 
                2, 
                color, 
                -1)

            if display_name:
                # Write the name of the landmark
                cv2.putText(
                    image_out,
                    landmark.n,
                    (landmark.x, landmark.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 ,
                    color
                )
        
        return image_out
