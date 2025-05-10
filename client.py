import cv2
import requests
import time
import psutil
import threading
from playsound import playsound
import os


SERVER_URL = "http://34.9.237.44:8000"
TELEGRAM_TOKEN = '7285124282:AAEL3q-2G5KxTZ8hB7a6Hq62E5jR0aVZ1TM'
TELEGRAM_CHAT_ID = '6510802773'

last_alert_time = 0
alert_cooldown = 5  # giây

# Gửi CPU và Memory usage
def send_metrics():
    while True:
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            requests.post(f"{SERVER_URL}/metrics", json={"cpu": cpu, "memory": mem})
        except Exception as e:
            print("Metric error:", e)
        time.sleep(1)

# Gửi hình ảnh từ webcam
def send_camera():
    global last_alert_time
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {
            'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')
        }
        try:
            response = requests.post(f"{SERVER_URL}/upload", files=files)
            if response.status_code == 200:
                print("Frame sent.")
                result = response.json()
                if result.get("fall_detected"):
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        last_alert_time = current_time
                        print("Té ngã! Phát âm thanh cảnh báo.")
                        threading.Thread(target=playsound, args=("alert.mp3",), daemon=True).start()
                        # threading.Thread(target=send_telegram_message, args=("⚠️ Cảnh báo: Phát hiện té ngã!",), daemon=True).start()
            else:
                print(f"Failed to send frame. Code: {response.status_code}")
        except Exception as e:
            print(f"Frame error: {e}")
        time.sleep(1/30)  # ~30fps

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message
        }
        requests.post(url, data=payload)
        print("Success tele")
    except Exception as e:
        print("Telegram error:", e)

if __name__ == "__main__":
    threading.Thread(target=send_metrics, daemon=True).start()
    send_camera()
