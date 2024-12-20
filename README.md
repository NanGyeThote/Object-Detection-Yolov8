# Object Detection and OpenAI Chat Application

This project is a Python-based application that integrates object detection using the YOLO model and interaction with OpenAI's GPT-4. It provides a graphical user interface (GUI) built with Tkinter to allow users to:

- Detect objects in images or videos.
- Perform live object detection through a connected camera.
- Interact with OpenAI's GPT-4 based on detected objects.

## Features

- **Object Detection:** Detects objects using a custom-trained YOLO model.
- **Video Support:** Detect objects in video files.
- **Live Camera Detection:** Perform real-time object detection from a webcam.
- **Image Analysis:** Load and analyze images for object detection.
- **OpenAI Integration:** Generate prompts and interact with GPT-4 based on detected objects.
- **User-Friendly Interface:** A simple GUI built with Tkinter.

## Requirements

The project requires the following Python packages:

- `opencv-python`
- `opencv-python-headless`
- `Pillow`
- `tk`
- `ultralytics`
- `cvzone`
- `numpy`
- `openai`

## Installation

1. Clone this repository or download the code.
2. Ensure Python 3.8 or later is installed on your system.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
    ```

4. Replace the placeholders in the code with your own trained YOLO model path (epoch35.pt) and OpenAI API key.

## Usage

1. Run the script:

    ```bash
    python main.py
    ```

2. Use the GUI to:

- **Select a video file for object detection.**.
- **Start live object detection using your webcam**.
- **Load an image for analysis**.

3. Load an image for analysis.

## Configuration

- **YOLO Model**: Replace epoch35.pt in the code with the path to your custom YOLO model.
- **OpenAI API Keys**: Replace 'your_api_keys' in the script with your OpenAI API key.
- **Class Name**: Update the classNames variable with the class names specific to your trained YOLO model.

## Control

- **Object Detection**:

    - **Press q to exit object detection mode while running**.

- **Close Application**:

    - **Use the Close button to exit the application**.

## Limitation

- **The project requires a trained YOLO model compatible with the ultralytics library**.
- **OpenAI API calls depend on your API quota and internet connectivity**.

## Future Developemnt

- **Add support for multiple object detection models**.
- **Implement advanced error handling and logging**.
- **Enhance the GUI for better user experience**.

## License

This project is licensed under the MIT License. See the LICENSE file for details. (Not Yet)

## Acknowlegements

- **[YOLO]() for object detection**.
- **[OpenAI]() for GPT-4 integration**.
- **[Computer vision engineer](https://youtu.be/PfQwNe0P-G4?si=pVWrjRFwhjrFmWCu) , here you can get explaination similar to my project**.
- **The Python community for creating robust libraries for development**.

## Contact 

For questions or suggestions, feel free to open an issue or contact me. You can read at my GitHub readme file.

    ```bash
    You can copy and paste this content into a `README.md` file. Let me know if you need further assistance!
    ```
