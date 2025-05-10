import time
import requests
import threading
import psutil
import io
from picamera2 import Picamera2, Preview
from gpiozero import Buzzer
sudo apt update
sudo apt install python3-picamera2 python3-gpiozero omxplayer


# Setup
SERVER_URL = "http://34.9.237.44:8000"
TELEGRAM_TOKEN = '7285124282:AAEL3q-2G5KxTZ8hB7a6Hq62E5jR0aVZ1TM'
TELEGRAM_CHAT_ID = '6510802773'

last_alert_time = 0
alert_cooldown = 5  # seconds

# Buzzer setup (assume connected to GPIO pin 17)
buzzer = Buzzer(17)

# Gửi CPU và RAM usage
def send_metrics():
    while True:
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            requests.post(f"{SERVER_URL}/metrics", json={"cpu": cpu, "memory": mem})
        except Exception as e:
            print("Metric error:", e)
        time.sleep(1)

# Phát cảnh báo bằng còi chip
def trigger_buzzer():
    print("Té ngã! Cảnh báo bằng còi chip.")
    buzzer.on()
    time.sleep(1)
    buzzer.off()

# Gửi hình từ Pi Camera V2
def send_camera():
    global last_alert_time
    picam = Picamera2()
    picam.configure(picam.create_still_configuration())
    picam.start()
    time.sleep(2)

    while True:
        try:
            frame = picam.capture_array()
            # encode to JPEG in memory
            _, jpeg = cv2.imencode('.jpg', frame)
            files = {
                'file': ('frame.jpg', jpeg.tobytes(), 'image/jpeg')
            }

            response = requests.post(f"{SERVER_URL}/upload", files=files)
            if response.status_code == 200:
                print("Frame sent.")
                result = response.json()
                if result.get("fall_detected"):
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        last_alert_time = current_time
                        threading.Thread(target=trigger_buzzer, daemon=True).start()
                        threading.Thread(target=send_telegram_message, args=("⚠️ Cảnh báo: Phát hiện té ngã!",), daemon=True).start()
            else:
                print(f"Failed to send frame. Code: {response.status_code}")
        except Exception as e:
            print(f"Frame error: {e}")
        time.sleep(1/10)  # khoảng 10fps

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message
        }
        requests.post(url, data=payload)
        print("Gửi Telegram thành công")
    except Exception as e:
        print("Telegram error:", e)

if __name__ == "__main__":
    threading.Thread(target=send_metrics, daemon=True).start()
    send_camera()
