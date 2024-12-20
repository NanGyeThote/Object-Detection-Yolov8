import cv2
import math
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cvzone
import openai
import numpy as np

# Initialize tkinter window
root = tk.Tk()
root.title("Object Detection and OpenAI Chat")

# Set initial window size
root.geometry("1000x800")  # Width x Height
model = YOLO('epoch35.pt') #Replace with your model

# Function to start object detection with selected video file
def start_detection_video():
    global cap, model, detected_objects, video_path

    video_path = filedialog.askopenfilename(initialdir="./videos", title="Select a Video File",
                                           filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))

    if video_path:
        cap = cv2.VideoCapture(video_path)
        model = YOLO('epoch35.pt') #Replace with your model
        detected_objects = []
        detect_objects()

# Function to start live object detection from camera
def start_detection_live():
    global cap, model, detected_objects

    cap = cv2.VideoCapture(0)  # 0 for default camera
    model = YOLO('epoch35.pt')  # Initialize YOLO model #Replace with your model
    detected_objects = []
    detect_objects()
    

#Your Open AI Keys
openai.api_key = 'your_api_keys' #Replace with your API Keys for chat response!

# Function to perform object detection
def detect_objects():
    global cap, model, detected_objects, classNames, output_text

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)
                update_detected_objects(f'{classNames[cls]}')

        cv2.imshow("Object Detection", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            video_capture_complete = True
            break
        if cv2.getWindowProperty('Object Detection', cv2.WND_PROP_VISIBLE) < 1:
            video_capture_complete = True
            break

    cap.release()
    cv2.destroyAllWindows()
    # Update output text
    if video_capture_complete:
        if detected_objects:
            detected_object = detected_objects[-1]
            prompt = generate_chat_prompt(detected_object)
            # chat_with_openai(prompt)
        else:
            update_detected_objects('No object Detected.') 
            output_text.set("No Objects")

# Function for chapGPT - prompt
# Function to generate prompt for ChatGPT based on detected object
def generate_chat_prompt(detected_object):
    if detected_object == 'person':
        return "Can you give me a nickname and greet me?"
    else:
        return f"What can I make with {detected_object}s? And what are the calories of {detected_object}s."

# Function to interact with OpenAI's GPT-4
def chat_with_openai(prompt):
    global output_text

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )

        output_text.set(response['choices'][0]['message']['content'].strip())
        #print(response['choices'][0]['message']['content'].strip())

    except openai.OpenAIError as e:
        output_text.set(f"OpenAI API Error: {e}")

# Function to update detected objects list
def update_detected_objects(name):
    detected_objects.append(name)

# Function to load and detect objects in a photo
def detect_objects_in_photo():
    global model, photo_label, output_text, classNames, detected_objects

    try:
        # Load photo file
        photo_path = filedialog.askopenfilename(initialdir="./", title="Select an Image File",
                                               filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.gif"), ("All files", "*.*")))
        if not photo_path:
            return

        # Load YOLO model if not already loaded
        if model is None:
            model = YOLO('epoch35.pt') #Replace with your model

        # Open photo using PIL
        image_pil = Image.open(photo_path)

        # Convert PIL image to OpenCV format (BGR)
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Perform object detection
        results = model(img_cv)

        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img_cv, (x1, y1, w, h))

                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])

                # Draw bounding box on the image
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Put label with class name and confidence
                label = f'{classNames[cls]} {conf}'
                cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Update detected objects list
                update_detected_objects(f'{classNames[cls]}')

        # Convert OpenCV image back to PIL Image for display
        image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image)

        # Update photo label in tkinter GUI
        if photo_label:
            photo_label.config(image=photo)
            photo_label.image = photo
        else:
            photo_label = tk.Label(root, image=photo)
            photo_label.pack(pady=20)

        # Update output text
        if detected_objects:
            detected_object = detected_objects[-1]
            #prompt = generate_chat_prompt(detected_object)
            #chat_with_openai(prompt)
            output_text.set(f"{detected_object}")
        else:
            update_detected_objects('No object Detected.')
            output_text.set("No Objects")

    except Exception as e:
        output_text.set(f"Error: {str(e)}")

# Close window function
def close_window():
    root.destroy()

# Initialize variables
cap = None
model = None
photo_label = None
detected_objects = []
classNames = ["Unripe Tomato", "Ripe Tomato"]  # Update with your class names
video_path = ""
image_label = None
output_text = tk.StringVar()

# GUI elements
start_video_button = tk.Button(root, text="Select Video File", command=start_detection_video, width=20, height=2)
start_video_button.pack(pady=20)

start_live_button = tk.Button(root, text="Start Live Capture", command=start_detection_live, width=20, height=2)
start_live_button.pack(pady=20)

load_image_button = tk.Button(root, text="Load Image", command=detect_objects_in_photo, width=20, height=2)
load_image_button.pack(pady=20)

output_label = tk.Label(root, textvariable=output_text, wraplength=500, justify="left")
output_label.pack(pady=20, padx=20)

# Close button
close_button = tk.Button(root, text="Close", command=close_window, width=20, height=2)
close_button.pack(pady=20)

# Run the tkinter main loop
root.mainloop()
