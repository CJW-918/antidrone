import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

model = YOLO('/home/raspberrypi/antidrone/best_n.pt')
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)

        if len(results.xyxy[0]) > 0:
            xmin = results.xyxy[0][0, 0].item()
            ymin = results.xyxy[0][0, 1].item()

            annotated_frame = frame.copy()
            cv2.circle(annotated_frame, (int(xmin), int(ymin)), 80, (0, 0, 255), 2)
            cv2.putText(annotated_frame, "DRONE DETECTED", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        else:
            annotated_frame = frame
            cv2.putText(annotated_frame, "NO DRONE DETECTED", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        cv2.imshow("Drone Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    picam2.close()
