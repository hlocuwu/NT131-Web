import cv2
import requests
import time

cap = cv2.VideoCapture(0)  # Mở camera

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    _, img_encoded = cv2.imencode('.jpg', frame)  # Mã hóa frame thành định dạng jpg
    
    try:
        # Gửi frame qua POST request
        response = requests.post("http://34.9.237.44:8000/upload", files={'file': img_encoded.tobytes()})
        
        if response.status_code == 200:
            print("Frame sent successfully.")
        else:
            print(f"Failed to send frame. Status code: {response.status_code}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Đặt delay để gửi frame theo tần suất hợp lý (ví dụ: 30 frame mỗi giây)
    time.sleep(0.033)  # Đặt thời gian delay khoảng 33ms (tương đương với 30fps)
