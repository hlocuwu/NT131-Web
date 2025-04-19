import cv2
import requests
import time
import psutil
import threading


SERVER_URL = "http://34.9.237.44:8000"

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
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        _, img_encoded = cv2.imencode('.jpg', frame)
        try:
            response = requests.post(f"{SERVER_URL}/upload", files={'file': img_encoded.tobytes()})
            if response.status_code == 200:
                print("Frame sent.")
            else:
                print(f"Failed to send frame. Code: {response.status_code}")
        except Exception as e:
            print(f"Frame error: {e}")
        time.sleep(0.033)  # ~30fps

if __name__ == "__main__":
    threading.Thread(target=send_metrics, daemon=True).start()
    send_camera()
