from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
# import threading # Không còn sử dụng trong code này
import time
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
# import collections
from google.cloud import storage
import json
from collections import defaultdict
from datetime import datetime

app = FastAPI()

# Cung cấp thư mục static
app.mount("/custom", StaticFiles(directory="custom"), name="custom")

# Load pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Giữ lại độ tin cậy để lọc bỏ phát hiện kém
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Shared states
latest_frame = None
fall_frame = None
last_fall_time = 0
metrics_data = {"cpu": 0, "memory": 0}

# --- Biến trạng thái cho phát hiện té ngã ---
# Không cần pose_history cho logic đơn giản
# is_falling = False # Không cần thiết cho logic đơn giản này
fall_cooldown = 5 # Giảm cooldown để test nhanh hơn (có thể chỉnh lại 10s)

# --- Hằng số cấu hình cho phát hiện té ngã ---
# Không cần ngưỡng vận tốc
# Không cần offset vị trí cuối
VISIBILITY_THRESHOLD = 0.6 # Giữ lại kiểm tra visibility

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
    global latest_frame, fall_frame, last_fall_time

    start_time = time.time()

    content = await file.read()
    np_img = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        print("Lỗi: Không thể decode ảnh")
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

        # Trigger phát hiện "té ngã" đơn giản mỗi khi thấy pose
        if current_time - last_fall_time > fall_cooldown:
            detected_fall_this_frame = True
            last_fall_time = current_time
            print("!!! PHÁT HIỆN POSE (Giả lập té ngã) !!!")

            _, jpeg_fall = cv2.imencode('.jpg', img)
            fall_frame = jpeg_fall.tobytes()
            log_fall_event_to_gcs(fall_frame, current_time)


    # Luôn cập nhật latest_frame để hiển thị stream chính
    _, jpeg = cv2.imencode('.jpg', img)
    latest_frame = jpeg.tobytes()

    processing_time = time.time() - start_time

    return {"message": "Image processed", "fall_detected": detected_fall_this_frame}



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
    stats_day = defaultdict(int)
    stats_week = defaultdict(int)
    stats_month = defaultdict(int)

    for blob in blobs:
        if blob.name.endswith(".json"):
            # Trích thời gian từ tên file
            try:
                timestamp_str = blob.name.split("/")[-1].replace(".json", "")
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

                # Ngày
                day_key = dt.strftime("%Y-%m-%d")
                stats_day[day_key] += 1

                # Tuần
                week_key = dt.strftime("%Y-W%U")
                stats_week[week_key] += 1

                # Tháng
                month_key = dt.strftime("%Y-%m")
                stats_month[month_key] += 1

            except Exception as e:
                print("Lỗi đọc timestamp:", blob.name, e)

    return {
        "day": dict(stats_day),
        "week": dict(stats_week),
        "month": dict(stats_month),
    }

if __name__ == "__main__":
    import uvicorn
    # Chạy với reload=False khi deploy hoặc nếu không cần tự động tải lại khi sửa code
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False) # Tắt reload để tránh reset state khi có request