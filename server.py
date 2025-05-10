from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import time
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
from google.cloud import storage
import json
from collections import defaultdict
from datetime import datetime
import calendar
from typing import Optional, Tuple, Dict, List
import math

app = FastAPI()
app.mount("/custom", StaticFiles(directory="custom"), name="custom")

# Pose detection setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
    )

# Shared states
latest_frame = None
fall_frame = None
last_fall_time = 0
metrics_data = {"cpu": 0, "memory": 0}
fall_cooldown = 5

# Fall detection parameters
VISIBILITY_THRESHOLD = 0.6
FALL_ANGLE_THRESHOLD = 45  # degrees
FALL_VELOCITY_THRESHOLD = 0.3  # normalized velocity
FALL_CONFIRMATION_FRAMES = 5  # number of consecutive frames to confirm fall
MIN_HEIGHT_RATIO = 0.4  # min height ratio to consider as fall

# Fall detection state
fall_detection_state = {
    "fall_counter": 0,
    "previous_landmarks": None,
    "previous_time": None,
    "confirmed_fall": False
}

class PoseAnalyzer:
    @staticmethod
    def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Tính góc giữa 3 điểm (tính bằng độ)"""
        ba = (a[0]-b[0], a[1]-b[1])
        bc = (c[0]-b[0], c[1]-b[1])
        
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        
        angle_rad = math.acos(dot_product / (mag_ba * mag_bc))
        return math.degrees(angle_rad)

    @staticmethod
    def calculate_velocity(prev_point: Tuple[float, float], curr_point: Tuple[float, float], time_diff: float) -> float:
        """Tính vận tốc di chuyển của điểm landmark"""
        if time_diff == 0:
            return 0
        dx = curr_point[0] - prev_point[0]
        dy = curr_point[1] - prev_point[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance / time_diff

    @staticmethod
    def get_landmark_coords(landmarks, idx: int) -> Optional[Tuple[float, float]]:
        """Lấy tọa độ của landmark nếu visibility đủ cao"""
        if landmarks.landmark[idx].visibility < VISIBILITY_THRESHOLD:
            return None
        return (landmarks.landmark[idx].x, landmarks.landmark[idx].y)

    @staticmethod
    def detect_fall(landmarks, previous_landmarks: Optional[mp_pose.PoseLandmark], 
                   previous_time: Optional[float]) -> Tuple[bool, Dict[str, float]]:
        """Phát hiện té ngã dựa trên các yếu tố: góc nghiêng, vận tốc, tỉ lệ chiều cao"""
        detection_metrics = {
            "torso_angle": 0,
            "velocity": 0,
            "height_ratio": 0,
            "is_falling": False
        }
        
        if landmarks is None:
            return False, detection_metrics
            
        # Lấy các điểm mốc quan trọng
        nose = PoseAnalyzer.get_landmark_coords(landmarks, mp_pose.PoseLandmark.NOSE)
        left_shoulder = PoseAnalyzer.get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = PoseAnalyzer.get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = PoseAnalyzer.get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = PoseAnalyzer.get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
        left_ankle = PoseAnalyzer.get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = PoseAnalyzer.get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        # Kiểm tra xem có đủ điểm mốc không
        if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
            return False, detection_metrics
        
        # Tính toán các giá trị trung bình
        shoulder_center = ((left_shoulder[0] + right_shoulder[0])/2, 
                          (left_shoulder[1] + right_shoulder[1])/2)
        hip_center = ((left_hip[0] + right_hip[0])/2, 
                     (left_hip[1] + right_hip[1])/2)
        
        # 1. Tính góc nghiêng của thân trên (so với phương thẳng đứng)
        vertical_angle = PoseAnalyzer.calculate_angle(
            (shoulder_center[0], shoulder_center[1] - 0.1),  # Điểm phía trên vai
            shoulder_center,
            hip_center
        )
        detection_metrics["torso_angle"] = vertical_angle
        
        # 2. Tính tỉ lệ chiều cao hiện tại so với chiều cao đứng
        if left_ankle and right_ankle:
            height_current = abs(shoulder_center[1] - (left_ankle[1] + right_ankle[1])/2)
            height_normal = abs(hip_center[1] - (left_ankle[1] + right_ankle[1])/2) * 2  # Ước lượng chiều cao đứng
            height_ratio = height_current / height_normal if height_normal > 0 else 1
            detection_metrics["height_ratio"] = height_ratio
        else:
            height_ratio = 1
        
        # 3. Tính vận tốc di chuyển của đầu (nếu có dữ liệu từ frame trước)
        velocity = 0
        current_time = time.time()
        if previous_landmarks and previous_time and current_time > previous_time:
            prev_nose = PoseAnalyzer.get_landmark_coords(previous_landmarks, mp_pose.PoseLandmark.NOSE)
            if prev_nose and nose:
                time_diff = current_time - previous_time
                velocity = PoseAnalyzer.calculate_velocity(prev_nose, nose, time_diff)
                detection_metrics["velocity"] = velocity
        
        # Kiểm tra điều kiện té ngã
        is_falling = (
            (vertical_angle > FALL_ANGLE_THRESHOLD) or  # Góc nghiêng lớn
            (height_ratio < MIN_HEIGHT_RATIO) or         # Chiều cao giảm đáng kể
            (velocity > FALL_VELOCITY_THRESHOLD)         # Di chuyển nhanh xuống
        )
        detection_metrics["is_falling"] = is_falling
        
        return is_falling, detection_metrics

def log_fall_event_to_gcs(image_bytes: bytes, timestamp: float):
    client = storage.Client()
    bucket = client.bucket("fall-log-data")  # Thay bằng tên bucket của bạn

    # Định dạng tên file theo timestamp
    readable_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(timestamp))
    image_filename = f"fall_events/{readable_time}.jpg"
    json_filename = f"fall_events/{readable_time}.json"

    # Upload ảnh
    image_blob = bucket.blob(image_filename)
    image_blob.upload_from_string(image_bytes, content_type="image/jpeg")

    # Tạo và upload metadata (json)
    event_info = {
        "event": "fall_detected",
        "timestamp": readable_time,
    }
    json_blob = bucket.blob(json_filename)
    json_blob.upload_from_string(json.dumps(event_info, indent=2), content_type="application/json")

    print(f"[GCS] Uploaded fall image and metadata at {readable_time}")

# Trang HTML (Giữ nguyên các hàm @app.get)
@app.get("/", response_class=HTMLResponse)
async def index():
    with open('templates/index.html') as f:
        return f.read()

@app.get("/camera", response_class=HTMLResponse)
async def camera_page():
    with open('templates/camera.html') as f:
        return f.read()

@app.get("/chart", response_class=HTMLResponse)
async def chart_page():
    with open('templates/chart.html') as f:
        return f.read()

@app.get("/setting", response_class=HTMLResponse)
async def setting_page():
    with open('templates/setting.html') as f:
        return f.read()

# Stream camera chính (Giữ nguyên)
def generate():
    global latest_frame
    while True:
        if latest_frame is not None: # Kiểm tra None rõ ràng hơn
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n'
            )
        # Chờ một chút để giảm tải CPU, nhưng vẫn đủ nhanh để stream mượt
        time.sleep(0.04) # ~25 FPS

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# === HÀM PHÁT HIỆN TÉ NGÃ ĐƠN GIẢN HÓA (Nằm xuống = Ngã) ===
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global latest_frame, fall_frame, last_fall_time, fall_detection_state

    start_time = time.time()
    content = await file.read()
    np_img = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"message": "Image decode error"}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    detected_fall_this_frame = False
    current_time = time.time()

    if results.pose_landmarks:
        # Vẽ landmarks lên ảnh
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # Phát hiện té ngã
        is_falling, metrics = PoseAnalyzer.detect_fall(
            results.pose_landmarks,
            fall_detection_state["previous_landmarks"],
            fall_detection_state["previous_time"]
        )

        # Cập nhật trạng thái phát hiện té ngã
        if is_falling:
            fall_detection_state["fall_counter"] += 1
            if fall_detection_state["fall_counter"] >= FALL_CONFIRMATION_FRAMES:
                fall_detection_state["confirmed_fall"] = True
        else:
            fall_detection_state["fall_counter"] = max(0, fall_detection_state["fall_counter"] - 1)
            fall_detection_state["confirmed_fall"] = False

        # Kiểm tra nếu phát hiện té ngã và đủ thời gian cooldown
        if (fall_detection_state["confirmed_fall"] and 
            current_time - last_fall_time > fall_cooldown):
            detected_fall_this_frame = True
            last_fall_time = current_time
            print(f"!!! PHÁT HIỆN TÉ NGÃ !!! (Góc: {metrics['torso_angle']:.1f}°, "
                  f"Vận tốc: {metrics['velocity']:.2f}, "
                  f"Tỉ lệ chiều cao: {metrics['height_ratio']:.2f})")

            # Lưu ảnh và ghi log
            _, jpeg_fall = cv2.imencode('.jpg', img)
            fall_frame = jpeg_fall.tobytes()
            log_fall_event_to_gcs(fall_frame, current_time)
            fall_detection_state["fall_counter"] = 0
            fall_detection_state["confirmed_fall"] = False

        # Cập nhật landmarks từ frame trước
        fall_detection_state["previous_landmarks"] = results.pose_landmarks
        fall_detection_state["previous_time"] = current_time

    # Cập nhật frame mới nhất
    _, jpeg = cv2.imencode('.jpg', img)
    latest_frame = jpeg.tobytes()

    processing_time = time.time() - start_time
    return {
        "message": "Image processed",
        "fall_detected": detected_fall_this_frame,
        "processing_time": processing_time
    }

# Trigger feed (ảnh khi té ngã) (Giữ nguyên)
@app.get("/trigger_feed")
async def trigger_feed():
    global fall_frame
    if fall_frame:
        # Trả về ảnh đã lưu trữ khi ngã/nằm
        return StreamingResponse(BytesIO(fall_frame), media_type="image/jpeg")
    else:
        # Trả về ảnh trống nếu chưa có cú ngã/nằm nào được ghi lại
        blank = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.putText(blank, "No Lie Down Yet", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        _, jpeg = cv2.imencode('.jpg', blank)
        return StreamingResponse(BytesIO(jpeg.tobytes()), media_type="image/jpeg")


# Nhận metrics (Giữ nguyên)
@app.post("/metrics")
async def receive_metrics(data: dict):
    metrics_data["cpu"] = data["cpu"]
    metrics_data["memory"] = data["memory"]
    return {"status": "received"}

@app.get("/get_metrics")
async def get_metrics():
    return metrics_data

@app.get("/fall_stats")
async def fall_stats():
    client = storage.Client()
    bucket = client.bucket("fall-log-data")  # thay bằng bucket của bạn

    blobs = bucket.list_blobs(prefix="fall_events/")
    
    # Tạo dictionary để lưu số lần té ngã theo giờ và theo ngày
    hourly_stats = defaultdict(int)
    daily_stats = defaultdict(int)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_month = datetime.now().strftime("%Y-%m")

    for blob in blobs:
        if blob.name.endswith(".json"):
            try:
                timestamp_str = blob.name.split("/")[-1].replace(".json", "")
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                
                # Chỉ lấy dữ liệu của ngày hiện tại cho thống kê giờ
                if dt.strftime("%Y-%m-%d") == current_date:
                    hour_key = dt.strftime("%H:00")
                    hourly_stats[hour_key] += 1
                
                # Chỉ lấy dữ liệu của tháng hiện tại cho thống kê ngày
                if dt.strftime("%Y-%m") == current_month:
                    day_key = dt.strftime("%Y-%m-%d")
                    daily_stats[day_key] += 1
                    
            except Exception as e:
                print("Lỗi đọc timestamp:", blob.name, e)

    # Đảm bảo có tất cả các giờ trong ngày (từ 00:00 đến 23:00)
    for hour in range(24):
        hour_key = f"{hour:02d}:00"
        if hour_key not in hourly_stats:
            hourly_stats[hour_key] = 0
    
    # Đảm bảo có tất cả các ngày trong tháng
    today = datetime.now()
    _, last_day = calendar.monthrange(today.year, today.month)
    for day in range(1, last_day + 1):
        day_key = f"{today.year}-{today.month:02d}-{day:02d}"
        if day_key not in daily_stats:
            daily_stats[day_key] = 0

    return {
        "hourly": dict(sorted(hourly_stats.items())),
        "daily": dict(sorted(daily_stats.items())),
    }

if __name__ == "__main__":
    import uvicorn
    # Chạy với reload=False khi deploy hoặc nếu không cần tự động tải lại khi sửa code
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False) # Tắt reload để tránh reset state khi có request