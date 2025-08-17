import cv2
import numpy as np
import time
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter, load_delegate
import cvzone
import pigpio

# 1. EdgeTPU 모델 로드
model_path = '/home/raspberrypi/cd antidrone/best_full_integer_quant_edgetpu.tflite'
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# 2. 카메라 초기화
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (640, 480)}
))
picam2.start()


# 3. pigpio 초기화
PAN_PIN, TILT_PIN = 17, 27
pi = pigpio.pi()
if not pi.connected:
    exit()

def set_angle(pin, angle):
    angle = max(0, min(180, angle))  # 안전 범위 제한
    pulse_width = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(pin, pulse_width)

# 초기값: 중앙(90도)
pan_angle, tilt_angle = 90, 90
set_angle(PAN_PIN, pan_angle)
set_angle(TILT_PIN, tilt_angle)


def preprocess(frame):
    input_shape = input_details[0]['shape']
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.uint8)
    return img

def postprocess(output_data):
    results = []
    for detection in output_data:
        x1, y1, x2, y2, score, class_id = detection
        if score > 0.5:  # 신뢰도 기준
            results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'score': score,
                'class_id': int(class_id)
            })
    return results

# 5. 메인 루프
try:
    start_time = time.time()
    frame_count = 0

    while True:
        frame = picam2.capture_array()
        input_data = preprocess(frame)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        detections = postprocess(np.squeeze(output_data))

        # 탐지 결과 처리
        if len(detections) > 0:
            det = detections[0]  # 가장 첫 번째 탐지만 사용
            x1, y1, x2, y2 = det['bbox']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 중심 좌표

            # 화면 중앙 (320, 240)과 비교
            dx = cx - 320
            dy = cy - 240

            # 픽셀 차이를 각도로 변환 (비율 조절 가능)
            pan_angle -= dx * 0.02
            tilt_angle += dy * 0.02

            set_angle(PAN_PIN, pan_angle)
            set_angle(TILT_PIN, tilt_angle)

            # 화면에 시각화
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cvzone.putTextRect(frame, f'Drone {det["score"]:.2f}', (x1, y1-10))
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
            cv2.circle(frame, (320, 240), 5, (255,0,0), -1)  # 중앙점 표시

        # FPS 표시
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cvzone.putTextRect(frame, f'FPS: {fps:.2f}', (10, 30))

        cv2.imshow("Drone Tracking - Coral TPU", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    picam2.close()
    pi.set_servo_pulsewidth(PAN_PIN, 0)
    pi.set_servo_pulsewidth(TILT_PIN, 0)
    pi.stop()
