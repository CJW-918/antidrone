from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# GPIO setup
laser_pin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(laser_pin, GPIO.OUT)

# Load YOLO model
model = YOLO('/home/raspberrypi/antidrone/best_s.pt')
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        annotated_frame = results[0].plot()  # Draw the bounding boxes on the frame

        if results and results[0].boxes:  # Check if there are any detections
            boxes = results[0].boxes

            for box in boxes:
                # Extract coordinates from the xyxy format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Convert tensor to numpy and integer
                
                # Calculate the center of the bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Check if the center is within a certain radius (50 pixels) from the image center (320, 240)
                if ((center_x - 320) ** 2 + (center_y - 240) ** 2) <= 900:
                    # If within the circle's range, activate the laser
                    GPIO.output(laser_pin, GPIO.HIGH)
                else:
                    GPIO.output(laser_pin, GPIO.LOW)

                # Draw additional elements (circle and crosshairs) on the annotated frame
                cv2.circle(annotated_frame, (320, 240), 30, (0, 255, 0), 2)
                cv2.line(annotated_frame, (0, 240), (640, 240), (255, 0, 0), 2)
                cv2.line(annotated_frame, (320, 0), (320, 480), (255, 0, 0), 2)

        cv2.imshow("Drone Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.close()
