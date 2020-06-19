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
        Initialize the model

        Args:
            model_xml_path (str): path to the model's structure in same folder as the .bin file
            device (str: 'CPU'): device to load the model on
            extensions_path (str | None: None): extensions to load in case it is needed
        '''
        self.conf_threshold = conf_threshold

        # Call the __init__ function from the parent class
        super().__init__(model_xml_path, device, extensions_path)

    def predict(self, image):
        '''
        Detect the head on the image
        
        Args:
        image (numpy.array BGR): image to do the inference on

        Returns:
        results (): 
        '''
        # Pre process the image
        input_image = self.preprocess_input(image)

        # Detect the heads
        output = self.inference(input_image)

        # Post process the output
        results = self.postprocess_output(output['detection_out'], image)

        return results

    def postprocess_output(self, output, image):
        '''
        Postprocss the output of the model

        Args:
            output (numpy.array): array [1, 1, #detections, 7] containing the 
                bbox of the detected heads.
            image (numpy.array): original image used for the inference
        
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
                label = int(detection[1])
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

