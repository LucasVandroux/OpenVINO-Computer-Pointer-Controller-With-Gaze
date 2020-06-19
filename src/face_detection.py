'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

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
        self.extensions_path = extension

    def load_model(self):
        """ Load the model 
        """        
        # Initialize the plugin
        self.plugin = IECore()

        # Read IR as a IENetwork
        self.network = IENetwork(model=self.model_structure_path, weights=self.model_weights_path)

        # Load the extensions that might be needed
        # No need of CPU extension with the 2020.3 version of OpenVINO
        if self.extensions_path is not None:
            plugin.add_extension(extension_path=self.extensions_path, device_name=self.device)
        
        # Check for supported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            sys.exit(f"[ ERR ] Unsupported layers found in model {self.model_name}: {unsupported_layers}.")

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, self.device)

        # Get the input layer and output layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

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
        input_image = preprocess_input(image)

        # Create the input dictionary
        input_dict = {self.input_blob: input_image}

        # Do the inference
        output = self.exec_network.infer(input_dict)

        # Post process the output
        results = postprocess_output(output)

        return results

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        raise NotImplementedError

    def postprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError
