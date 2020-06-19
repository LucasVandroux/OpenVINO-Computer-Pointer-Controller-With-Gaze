import os
import sys
import time

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore


class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_xml_path, device='CPU', extensions_path=None):
        '''
        Initialize the model

        Args:
        model_xml_path (str): path to the model's structure in same folder as the .bin file
        device (str: 'CPU'): device to load the model on
        extensions_path (str | None: None): extensions to load in case it is needed
        '''
        # Extract the path to the directory containing the models files
        self.model_dir_path, filename = os.path.split(model_xml_path)
        
        # Extract the name of the model
        self.model_name = filename.split('.')[0]
        
        # Create the path to the .xml file describing the structure of the model
        self.model_structure_path = os.path.join(self.model_dir_path, self.model_name + '.xml')
       
        # Create the path to the .bin file containing the weights of the model
        self.model_weights_path = os.path.join(self.model_dir_path, self.model_name + '.bin')
        
        self.device = device
        self.extensions_path = extensions_path

    def load_model(self):
        """ Load the model 
        """
        # Initialize timer
        start_time = time.time()

        # Initialize the plugin
        self.plugin = IECore()

        # Read IR as a IENetwork
        self.network = IENetwork(model=self.model_structure_path, weights=self.model_weights_path)

        # Load the extensions that might be needed
        # No need of CPU extension with the 2020.3 version of OpenVINO
        if self.extensions_path is not None:
            self.plugin.add_extension(extension_path=self.extensions_path, device_name=self.device)
        
        # Check for supported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            sys.exit(f"[ ERRO ] Unsupported layers found in model {self.model_name}: {unsupported_layers}.")

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, self.device)

        # Get the input layer and output layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        print(f"[ INFO ] {self.model_name} loaded in {time.time() - start_time}s.")

    def get_input_shape(self):
        """ Gets the input shape of the network
        """
        return self.network.inputs[self.input_blob].shape

    def predict(self, image):
        '''
        Do the prediction on an image
        
        Args:
        image (numpy.array BGR): image to do the inference on

        Returns:
        results (): 
        '''
        # Pre process the image
        input_image = self.preprocess_input(image)

        # Create the input dictionary
        input_dict = {self.input_blob: input_image}

        # Do the inference
        output = self.exec_network.infer(input_dict)

        # Post process the output
        results = self.postprocess_output(output)

        return results

    def preprocess_input(self, image):
        '''
        Preprocess the image

        Args:
        image (numpy.array BGR): image to preprocess

        Returns:
        input_image (numpy.array): array containing the image with the dimensions [1x3x384x672]
        '''
        # Get the input shape of the model
        net_input_shape = self.get_input_shape()

        # Resize the image to the input of the model
        input_image = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))

        # Get the Color Channels in front
        input_image = input_image.transpose((2,0,1))

        # Add one dimension at the front for the batch size
        input_image = input_image.reshape(1, *input_image.shape)

        return input_image

    def postprocess_output(self, outputs):
        '''
        Postprocss the output of the model

        Args:
        outputs (numpy.array): array of the dimensions [1, 1, #detections, 7]

        Returns:
        results (list):
        '''
        num_detections = outputs.shape[2]
        results = []

        for idx in range(num_detections):
            detection = outputs[1,1,idx,:]
            results.append({
                'label': detection[1],
                'conf': detection[2],
                'x_min': detection[3],
                'y_min': detection[4],
                'x_max': detection[5],
                'y_max': detection[6]
            })

        # Order results with biggest conf first
        sorted(results, key = lambda i: i['conf'], reverse=True)

        return results

