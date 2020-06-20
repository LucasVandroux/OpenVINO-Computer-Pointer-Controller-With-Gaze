# Computer Pointer Controller

The project aims at controlling your mouse on your computer screen by using your gaze. For this we use a computer vision pipeline of four different deep learning models to extract the necessary information to infere the direction of your gaze.

## Installation
We recommend using an isolated Python environment, such as [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/) with at least **Python 3.6**. Then, use the following lines of code:

```
# Clone the repository
git clone https://github.com/LucasVandroux/OpenVINO-Computer-Pointer-Controller-With-Gaze.git
cd OpenVINO-Computer-Pointer-Controller-With-Gaze

# Create a python virtual environment to install all the needed packages
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

# Initialized the OpenVINO environment
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6

# Download the models
python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001 --precisions FP32-INT1
python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001 --precisions FP16-INT8
python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009 --precisions FP16-INT8
python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002 --precisions FP16-INT8
```

## Demo
To run the demo, simply run the following command from the root folder:
```
python src/main.py -i bin/demo.mp4
```

To only display the outputs of the different models, use the following command:
```
python src/main.py -i bin/demo.mp4 --display_outputs --disable_pointer_controller
```

## Repository Structure

The repository contains three different directories:
- **bin** : this directory contains all the binaries of the files used for testing such as the `demo.mp4`.
- **src** : this directory contains all the python code needed to run the demo.
- **intel** : if you used the instructions to download the models from the OpenVINO open model zoo, then you will have this directory containing all the downloaded models.


## Documentation
The main script offers different parameters to control the demo:

| name | type | default | description |
|------|------|---------|-------------|
|`--input` | str | **REQUIRED**     | Path to an image or video file. Otherwise, use CAM to select the webcam.|
| `--device`| str | CPU |Specify the target device to infer on: "CPU, GPU, FPGA or MYRIAD is acceptable. |
| `--cpu_extension` | str | None | MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl. |
| `--model_face_detection` | str | intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml | Path to the xml file for the face detection model. |
| `--model_head_pose` | str | intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml | Path to the xml file for the head pose model. |
| `--model_face_landmark` | str | intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml | Path to the xml file for the face landmark model. |
| `--model_gaze_estimation` | str | intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml | Path to the xml file for the head pose model. |
| `--display_outputs` | bool | False | Display the outputs of the models. |
| `--disable_pointer_controller` | bool | False | Disable the control of the pointer. |

You can get more information using the command:

```
python src/main.py --help
```

## Benchmarks
We benchmarked the average FPS of the system and the size of the models using different model precisions:

| Precision   | Average FPS | Size (MB) |
|-------------|-------------|-----------|
| `FP32`      | 63.27       | 17.18     |
| `FP16`      | 77.85       | 9.63      |
| `FP16-INT8` | 81.41       | 6.36      |


**N.B.** The face detection model is only available with the precision FP32-INT1. Therefore the precision in table above only affects the three other models.

## Results
As seen in the benchmarks section, the `FP16-INT8` precision is the fastest and the smallest in size. Therefore, for the purpose of this demo, we recommend to use the `FP16-INT8` precision for all the models except the face detection one that only has the precision `PF32-INT1`.

### Edge Cases
For the moment the demo only analyse a single face on the video. In case multiple persons's face are detected, only the one with the highest confidence is processed further.

Additionally, depending on the pose of the head, an eye might be not visible. We could stop the tracking if we detect the head pose angles are above a certain threshold.