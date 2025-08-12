import cv2
import numpy as np
import pandas as pd
import time
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import cvzone

# Edge TPU용 TFLite 모델 경로
model_path = '/home/raspberrypi/cd antidrone/best_full_integer_quant_edgetpu.tflite'

# Edge TPU delegate를 명시하여 인터프리터 생성
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()  # 텐서 할당

# 입력, 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Picamera2 초기화 및 설정 (640x480, BGR888 포맷)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
picam2.start()

frame_count = 0
start_time = time.time()

def preprocess(frame):
    # 모델 입력 크기에 맞게 리사이즈 (예: 640x640)
    input_shape = input_details[0]['shape']  # 보통 [1, 높이, 너비, 3]
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    # BGR -> RGB 변환 (모델 입력 형식 맞춤)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 모델이 uint8 타입일 경우, 타입 변환 및 배치 차원 추가
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.uint8)
    return img

def postprocess(output_data):
    # 모델 출력 결과 후처리
    # (출력 형식에 따라 맞게 수정 필요)
    # 예: [x1, y1, x2, y2, score, class_id] 형태 배열이라 가정

    results = []
    for detection in output_data:
        x1, y1, x2, y2, score, class_id = detection
        if score > 0.3:  # 신뢰도 임계값 설정
            results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'score': score,
                'class_id': int(class_id)
            })
    return results

while True:
    frame = picam2.capture_array()  # 카메라에서 프레임 캡처
    input_data = preprocess(frame)  # 전처리

    interpreter.set_tensor(input_details[0]['index'], input_data)  # 입력 텐서 설정
    interpreter.invoke()  # 추론 실행

    output_data = interpreter.get_tensor(output_details[0]['index'])  # 출력 결과 가져오기
    output_data = np.squeeze(output_data)  # 불필요한 차원 제거
    detections = postprocess(output_data)  # 후처리로 바운딩박스 추출

    # 원본 프레임에 바운딩 박스 및 텍스트 그리기
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)  # 초록색 사각형
        cvzone.putTextRect(frame, f'Drone {score:.2f}', (x1, y1-10), scale=1, thickness=1)  # 라벨 표시

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time  # FPS 계산
    cvzone.putTextRect(frame, f'FPS: {fps:.2f}', (10, 30), scale=1, thickness=1)  # FPS 표시

    cv2.imshow("Drone Detection - Coral TPU", frame)  # 화면에 출력

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()