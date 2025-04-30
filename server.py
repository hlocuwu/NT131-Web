from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import threading
import time
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO

app = FastAPI()

# Cung cấp thư mục static
app.mount("/custom", StaticFiles(directory="custom"), name="custom")

# Đọc và render HTML templates
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

# Tạo MJPEG stream
latest_frame = None
def generate():
    global latest_frame
    while True:
        if latest_frame:
            frame = latest_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# API nhận ảnh từ laptop
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global latest_frame, fall_frame, last_fall_time

    content = await file.read()
    latest_frame = content

    # Decode ảnh
    np_img = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Chạy MediaPipe Pose
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Logic phát hiện té ngã (đơn giản: góc lưng ngang quá thấp)
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        avg_hip_y = (left_hip.y + right_hip.y) / 2

        # Điều kiện đơn giản: nếu đầu (mũi) thấp hơn hông → té
        if nose.y > avg_hip_y:
            current_time = time.time()
            if current_time - last_fall_time > 10:  # mỗi 10s lưu lại 1 lần
                # Encode ảnh lại
                _, jpeg = cv2.imencode('.jpg', img)
                fall_frame = jpeg.tobytes()
                last_fall_time = current_time
                print("Phát hiện té ngã!")
    
    return {"message": "Image processed"}

# Nhận và lưu dữ liệu CPU / Memory từ client
metrics_data = {"cpu": 0, "memory": 0}

@app.post("/metrics")
async def receive_metrics(data: dict):
    metrics_data["cpu"] = data["cpu"]
    metrics_data["memory"] = data["memory"]
    return {"status": "received"}

@app.get("/get_metrics")
async def get_metrics():
    return metrics_data

@app.get("/trigger_feed")
async def trigger_feed():
    global fall_frame
    if fall_frame:
        return StreamingResponse(BytesIO(fall_frame), media_type="image/jpeg")
    else:
        # Trả về ảnh trống (hoặc placeholder nếu muốn)
        blank = np.zeros((200, 300, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', blank)
        return StreamingResponse(BytesIO(jpeg.tobytes()), media_type="image/jpeg")


# AI
latest_frame = None
fall_frame = None
last_fall_time = 0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


