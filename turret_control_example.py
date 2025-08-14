import pigpio # considering to use pigpio library to precise motor control
import time

# GPIO 핀 설정
PAN_PIN = 17   # 좌우
TILT_PIN = 27  # 상하

# pigpio 초기화
pi = pigpio.pi()
if not pi.connected:
    exit()

# 서보 각도 → 펄스폭 변환 함수
def set_angle(pin, angle):
    # SG90: 0도 ≈ 500µs, 180도 ≈ 2500µs
    pulse_width = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(pin, pulse_width)

try:
    # init location (middle)
    set_angle(PAN_PIN, 90)
    set_angle(TILT_PIN, 90)
    time.sleep(1)

    # test: left right
    for angle in range(60, 121, 5):
        set_angle(PAN_PIN, angle)
        time.sleep(0.05)
    for angle in range(120, 59, -5):
        set_angle(PAN_PIN, angle)
        time.sleep(0.05)

    # test: up down
    for angle in range(60, 121, 5):
        set_angle(TILT_PIN, angle)
        time.sleep(0.05)
    for angle in range(120, 59, -5):
        set_angle(TILT_PIN, angle)
        time.sleep(0.05)

finally:
    # 서보 PWM OFF (전원 차단)
    pi.set_servo_pulsewidth(PAN_PIN, 0)
    pi.set_servo_pulsewidth(TILT_PIN, 0)
    pi.stop()
