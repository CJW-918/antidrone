import cv2
import numpy as np
import time
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter, load_delegate
import pigpio

# 실행 옵션 (속도 ↑: False로)
DRAW_OVERLAY = True    # 박스/텍스트 그리기
SHOW_WINDOW  = True    # imshow 창 출력
UPDATE_EVERY_N_FRAMES = 1  # 서보 업데이트/오버레이 갱신 주기

# OpenCV 내부 스레드 수 (라즈베리파이에서 1이 더 안정/빠른 경우가 많음)
cv2.setNumThreads(1)

# ==============================
# 1) EdgeTPU 모델 로드
# ==============================
model_path = '/home/raspberrypi/cd antidrone/best_full_integer_quant_edgetpu.tflite'
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 텐서 shape (1, H, W, 3) 가정
in_batch, in_h, in_w, in_c = input_details[0]['shape']
assert in_c == 3, "모델 입력 채널이 3이 아닙니다."

# ==============================
# 2) 카메라 초기화 (640x480 고정)
# ==============================
FRAME_W, FRAME_H = 640, 480
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (FRAME_W, FRAME_H)}
    )
)
picam2.start()

# ==============================
# 3) pigpio 초기화
# ==============================
PAN_PIN, TILT_PIN = 17, 27
pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpio 데몬에 연결 실패. `sudo pigpiod` 실행을 확인하세요.")

def set_angle(pin, angle):
    # 범위 제한 + 펄스폭 변환 (SG90 기준)
    angle = 0 if angle < 0 else (180 if angle > 180 else angle)
    pulse_width = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(pin, pulse_width)

# 초기값: 중앙
pan_angle, tilt_angle = 90.0, 90.0
set_angle(PAN_PIN, pan_angle)
set_angle(TILT_PIN, tilt_angle)


# 4) 사전 할당 버퍼 (복사 최소화)
# 입력용 버퍼 (TFLite에 넣을 최종 RGB)
input_buffer = np.empty((1, in_h, in_w, 3), dtype=np.uint8)
# 중간 버퍼: resize 결과(BGR), 그걸 RGB로 변환한 결과
resized_bgr = np.empty((in_h, in_w, 3), dtype=np.uint8)
resized_rgb = np.empty((in_h, in_w, 3), dtype=np.uint8)

# 텍스트 그리기용 설정 (오버레이 True일 때만 사용)
font = cv2.FONT_HERSHEY_SIMPLEX


# 5) 후처리 함수
# 출력 형식: [N, 6] == [x1,y1,x2,y2,score,class]
def postprocess(output_data, score_thr=0.5):
    results = []
    # output_data는 squeeze되어 [N,6] 가정
    # 메모리 접근 최소화를 위해 Python loop는 유지하되 필터링은 단순화
    for det in output_data:
        score = det[4]
        if score > score_thr:
            x1 = int(det[0]); y1 = int(det[1]); x2 = int(det[2]); y2 = int(det[3])
            results.append({
                'bbox': [x1, y1, x2, y2],
                'score': float(score),
                'class_id': int(det[5])
            })
    return results


# 6) 메인 루프
try:
    start = time.time()
    frame_count = 0
    last_servo_update = 0

    while True:
        frame = picam2.capture_array()  # BGR (640x480)

        # 전처리 (사전할당 버퍼 사용)
        # resize: BGR → resized_bgr (dst 사용으로 메모리 재할당 방지)
        cv2.resize(frame, (in_w, in_h), resized_bgr, interpolation=cv2.INTER_LINEAR)
        # BGR → RGB (dst 사용)
        cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB, dst=resized_rgb)
        # 배치 차원에 넣기 (copy 없이 view 할당)
        input_buffer[0, ...] = resized_rgb

        # 추론
        interpreter.set_tensor(input_details[0]['index'], input_buffer)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.squeeze(output_data)  # [N,6]

        detections = postprocess(output_data, score_thr=0.5)

        # 최상 score 1개 선택 후 제어
        if detections:
            det = max(detections, key=lambda d: d['score'])
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 주기적으로만 서보 업데이트
            if frame_count - last_servo_update >= UPDATE_EVERY_N_FRAMES:
                dx = cx - CENTER_X
                dy = cy - CENTER_Y

                # 간단한 비례 제어 (게인 낮추면 진동 ↓)
                Kp = 0.02
                pan_angle -= dx * Kp
                tilt_angle += dy * Kp

                set_angle(PAN_PIN, pan_angle)
                set_angle(TILT_PIN, tilt_angle)
                last_servo_update = frame_count

            if DRAW_OVERLAY:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                label = f"Drone {det['score']:.2f}"
                cv2.putText(frame, label, (x1, max(0, y1-8)), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

        # FPS 계산
        frame_count += 1
        if DRAW_OVERLAY:
            elapsed = time.time() - start
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), font, 0.6, (255,255,255), 1, cv2.LINE_AA)

            # 중앙점 표시
            cv2.circle(frame, (CENTER_X, CENTER_Y), 3, (255,0,0), -1)

        if SHOW_WINDOW:
            cv2.imshow("Drone Tracking - Coral TPU (fast)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # headless 모드: 간단한 종료 조건만
            if frame_count % 3000 == 0:
                pass

finally:
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    picam2.close()
    # 서보 PWM OFF
    pi.set_servo_pulsewidth(PAN_PIN, 0)
    pi.set_servo_pulsewidth(TILT_PIN, 0)
    pi.stop()
