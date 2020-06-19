import cv2

from model import OpenVINOModel
from utils import BoundingBox

'''
Documentation: https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html

Download: python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001 --precisions FP32-INT1
'''

class FaceDetectionModel(OpenVINOModel):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_xml_path, device='CPU', extensions_path=None, conf_threshold = 0.8):
        '''
        Initialize the model.

        Args:
            model_xml_path (str): path to the model's structure in same folder as the .bin file.
            device (str: 'CPU'): device to load the model on.
            extensions_path (str | None: None): extensions to load in case it is needed.
        '''
        self.conf_threshold = conf_threshold

        # Call the __init__ function from the parent class
        super().__init__(model_xml_path, device, extensions_path)

    def predict(self, image):
        '''
        Detect the head on the image.
        
        Args:
            image (numpy.array BGR): image to do the inference on.

        Returns:
            results (list[BoundingBox]): list of the detected heads above
                the conf_threshold.
        '''
        # Pre process the image
        input_image = self.preprocess_input(image)

        # Detect the heads
        output = self.inference(input_image)

        # Post process the output
        results = self.postprocess_output(output[self.output_blob], image)

        return results

    def postprocess_output(self, output, image):
        '''
        Postprocss the output of the model.

        Args:
            output (numpy.array): array [1, 1, #detections, 7] containing the 
                bbox of the detected heads.
            image (numpy.array): original image used for the inference.
        
        Returns:
            detected_heads (list[BoundingBox]): list of the detected heads above
                the conf_threshold. 
        '''
        # Extract the dimension on the input image
        image_h, image_w, _ = image.shape

        # Extract the total number of detections
        num_detections = output.shape[2]

        # Initialize the list to store the detected heads
        detected_heads = []

        # Extract the detections
        for idx in range(num_detections):
            detection = output[0,0,idx,:]
            conf = detection[2]

            # Check if the confidence score is above the detection threshold
            if conf > self.conf_threshold:
                label = 'head'
                x_min = int(detection[3] * image_w)
                y_min = int(detection[4] * image_h)
                width = int(detection[5] * image_w - x_min)
                height = int(detection[6] * image_h - y_min)

                # Add the detection to the list of detected heads
                detected_heads.append(
                    BoundingBox(label, conf, x_min, y_min, width, height)
                )

        # Order results with biggest conf first
        sorted(detected_heads, key = lambda bbox: bbox.c, reverse=True)

        return detected_heads

    def display_output(self, image, results, color = (0, 255, 0)):
        '''
        Display the bounding boxes on the image.

        Args:
            image (numpy.array): original image used for the inference.
            results (list[BoundingBox]): list of the detected heads above
                the conf_threshold.
            color ((B, G, R): Green): color to draw the bounding boxes

        Returns:
            image_out (numpy.array): copy of the input image with the bounding 
                boxes of the detected head
        '''
        # Copy the input image
        image_out = image.copy()

        for detection in results:
            # Draw the bounding box
            cv2.rectangle(
                image_out,
                (detection.x, detection.y),
                (detection.x + detection.w, detection.y + detection.h),
                color,
                2,
            )

            # Write the confidence score
            cv2.putText(
                image_out,
                str(round(detection.c, 2)),
                (detection.x, detection.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 ,
                color
            )
        
        return image_out

