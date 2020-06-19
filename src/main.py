from argparse import ArgumentParser
import sys
import time

from input_feeder import InputFeeder
from face_detection import FaceDetectionModel

# Name of the cv2 window to display the feed
WINDOW_NAME = 'Computer Pointer Controller With Gaze'

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to an image or video file. Otherwise, use CAM to select the webcam.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable.")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("--model_face_detection", type=str,
                        default='TODO',
                        help="Path to the xml file for the face detection model.")
    parser.add_argument("--model_head_pose", type=str,
                        default='TODO',
                        help="Path to the xml file for the head pose model.")
    parser.add_argument("--model_face_landmark", type=str,
                        default='TODO',
                        help="Path to the xml file for the face landmark model.")
    parser.add_argument("--model_gaze_estimation", type=str,
                        default='TODO',
                        help="Path to the xml file for the head pose model.")
    return parser

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    # --- INPUT ---
    # Create a flag for single images
    image_flag = False

    # Initialize the input_type
    input_type = None
    
    # Check if the input is a webcam
    if args.input == 'CAM':
        input_type = 'cam'
    
    # Check if the input is an image
    elif args.input.endswith(('.jpg', '.bmp', '.png')):
        input_type = 'image'

    # Check if the input is a video
    elif args.input.endswith(('.mp4', '.avi')):
        input_type = 'video'

    else:
        sys.exit(f"[ ERRO ] The format of the input file '{args.input.endswith}' is not supported.")

    # Initialize the InputFeeder
    input_feeder = InputFeeder(input_type, args.input)
    input_feeder.load_data()

    # --- MODELS ---
    face_detection_model = FaceDetectionModel(
        model_xml_path = args.model_face_detection,
        device = args.device,
        extensions_path = args.cpu_extension,
    )

    face_detection_model.load_model()

    # --- WINDOW ---
    # Set the window to fullscreen
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    #Loop until stream is over
    while True:
        # start the timer
        start_time = time.time()

        # Read from the video capture
        frame = input_feeder.next_batch()

        # Detect the head on the frame
        list_heads = face_detection_model.predict(frame)

        print(list_heads)

        # Calculate and print the FPS
        fps = round(1/(time.time() - start_time), 2)
        cv2.rectangle(frame, (10, 2), (50,20), (255,255,255), -1)
        cv2.putText(frame, f"{fps}FPS",(15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        # Display the frame
        cv2.imshow(WINDOW_NAME, frame)

        # Wait for 'ESC' or 'q' to exit the program
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    # Release the input feeder
    input_feeder.close()

    # Destroy any OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Grab command line args
    args = build_argparser().parse_args()

    # Perform inference on the input stream
    infer_on_stream(args)
