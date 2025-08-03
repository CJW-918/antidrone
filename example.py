import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time
from picamera2 import Picamera2, Preview

# Load the YOLO model (optimized for Edge TPU)
model = YOLO('/home/raspberrypi/cd antidrone/best_full_integer_quant_edgetpu.tflite')

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
picam2.start()

frame_count = 0
start_time = time.time()

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for YOLO model input
    frame_resized = cv2.resize(frame, (640, 640))
    frame_count += 1
    if frame_count % 3 != 0:  # Skip every 3rd frame to optimize performance
        continue

    # Perform object detection using the YOLO model (with image size 640x640)
    results = model.predict(frame, imgsz=640)
    a = results[0].boxes.data.cpu().numpy()  # Get bounding box data for detected objects
    px = pd.DataFrame(a).astype("float")  # Convert bounding box data to a Pandas DataFrame

    # Loop through each detected object and draw bounding boxes and labels
    for index, row in px.iterrows():
        x1 = int(row[0])  # Bounding box top-left x-coordinate
        y1 = int(row[1])  # Bounding box top-left y-coordinate
        x2 = int(row[2])  # Bounding box bottom-right x-coordinate
        y2 = int(row[3])  # Bounding box bottom-right y-coordinate
        conf = row[4]  # Confidence score

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Display "Drone" and the confidence score
        cvzone.putTextRect(frame, f'Drone {conf:.2f}', (x1, y1), 1, 1)

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Display FPS on the frame
    cvzone.putTextRect(frame, f'FPS: {round(fps, 2)}', (10, 30), 1, 1)

    # Show the frame on the screen
    cv2.imshow("Drone Detection", frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
picam2.close()
