from ultralytics import YOLO
import cv2
from picamera2 import Picamera2, Preview

model = YOLO('/home/raspberrypi/antidrone/best_n.pt') # specify model to use
picam2 = Picamera2() #initialize camera
# set the frame
picam2.configure(picam2.create_preview_configuration(main={"format":"BGR888", "size":(640, 480)}))
picam2.start() #start camera

while True:
	# capture a frame from camera.
	frame = picam2.capture_array()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# use model to detect object in captured frame
	results = model(frame)
	#annotate the detected objects on the frame
	annotated_frame = results[0].plot()
	#display annotated frame with title 'Drone Detection'
	cv2.imshow("Drone Detection", annotated_frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #end detection by press 'q'
		break
		
cv2.destroyAllWindows() # end camera
picam2.close()
