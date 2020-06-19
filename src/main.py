from argparse import ArgumentParser
import sys
import time

import cv2 

from input_feeder import InputFeeder

from face_detection import FaceDetectionModel
from head_pose_estimation import HeadPoseEstimationModel
from facial_landmarks_detection import FacialLandmarksDetectionModel

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
                        default='intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml',
                        help="Path to the xml file for the face detection model.")
    parser.add_argument("--model_head_pose", type=str,
                        default='intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml',
                        help="Path to the xml file for the head pose model.")
    parser.add_argument("--model_face_landmark", type=str,
                        default='intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml',
                        help="Path to the xml file for the face landmark model.")
    parser.add_argument("--model_gaze_estimation", type=str,
                        default='TODO',
                        help="Path to the xml file for the head pose model.")
    parser.add_argument("--display_outputs", action="store_true",
                        help="Display the outputs of the models.")
    return parser

def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    # --- INPUT ---
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
    # Load the Face Detection Model
    face_detection_model = FaceDetectionModel(
        model_xml_path = args.model_face_detection,
        device = args.device,
        extensions_path = args.cpu_extension,
    )

    face_detection_model.load_model()

    # Load the Head Pose Estimation Model
    head_pose_estimation_model = HeadPoseEstimationModel(
        model_xml_path = args.model_head_pose,
        device = args.device,
        extensions_path = args.cpu_extension,
    )

    head_pose_estimation_model.load_model()

    # Load the Facial Landmarks Detection Model
    facial_landmarks_detection_model = FacialLandmarksDetectionModel(
        model_xml_path = args.model_face_landmark,
        device = args.device,
        extensions_path = args.cpu_extension,
    )

    facial_landmarks_detection_model.load_model()

    # --- WINDOW ---
    # Set the window to fullscreen
    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    #Loop until stream is over
    for frame in input_feeder.next_batch():
        # If there is no frame break the loop
        if frame is None:
            break

        # start the timer
        start_time = time.time()

        # Initialize the frame to be displayed
        display_frame = frame

        # --- DETECT HEAD ---
        # Detect the head on the frame
        list_heads = face_detection_model.predict(frame)

        # Draw the outputs of the head detection algorithm
        if args.display_outputs:
             display_frame = face_detection_model.display_output(frame, list_heads)

        # --- HEAD POSE ESTIMATION ---
        # Extract the roi of the head with the highest confidence score
        head = list_heads[0]
        head_x_max = head.x + head.w
        head_y_max = head.y + head.h

        head_roi = frame[head.y:head_y_max, head.x:head_x_max, :]

        # Estimate the pose of the best head
        head_angles = head_pose_estimation_model.predict(head_roi)

        # Draw the pose of the best head
        if args.display_outputs:
            display_head_pose = head_pose_estimation_model.display_output(head_roi, head_angles)
            display_frame[head.y:head_y_max, head.x:head_x_max, :] = display_head_pose

        # --- FACIAL LANDMARKS DETECTION ---
        # Detect the facial landmarks on the head with the highest confidence score
        face_landmarks = facial_landmarks_detection_model.predict(head_roi)

        # Draw the facial landmarks of the best head
        if args.display_outputs:
            # Set display_name to True to display the name of the landmarks
            display_facial_landmarks = facial_landmarks_detection_model.display_output(display_head_pose, face_landmarks, display_name = True)
            display_frame[head.y:head_y_max, head.x:head_x_max, :] = display_facial_landmarks


        # Calculate and print the FPS
        fps = round(1/(time.time() - start_time), 2)
        cv2.rectangle(display_frame, (10, 2), (120,20), (255,255,255), -1)
        cv2.putText(display_frame, f"{fps} FPS",(15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        # Display the frame
        cv2.imshow(WINDOW_NAME, display_frame)

        # Wait for 'ESC' or 'q' to exit the program
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    # Release the input feeder
    input_feeder.close()

    # Destroy any OpenCV windows
    cv2.destroyAllWindows()

    print(f"[ INFO ] Successfully exited the program.")

if __name__ == '__main__':
    # Grab command line args
    args = build_argparser().parse_args()

    # Perform inference on the input stream
    infer_on_stream(args)

