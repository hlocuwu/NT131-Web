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

# Load pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Shared states
latest_frame = None
fall_frame = None
last_fall_time = 0
metrics_data = {"cpu": 0, "memory": 0}

# Trang HTML
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

# Stream camera chính
def generate():
    global latest_frame
    while True:
        if latest_frame:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n'
            )
        time.sleep(0.05)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# Nhận ảnh từ client
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global latest_frame, fall_frame, last_fall_time

    content = await file.read()
    np_img = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        avg_hip_y = (left_hip.y + right_hip.y) / 2

        print(f"nose.y: {nose.y}, avg_hip_y: {avg_hip_y}")  # Debug

        if nose.y > avg_hip_y:
            current_time = time.time()
            if current_time - last_fall_time > 10:
                _, jpeg_fall = cv2.imencode('.jpg', img)
                fall_frame = jpeg_fall.tobytes()
                last_fall_time = current_time
                print("Phát hiện té ngã!")

    _, jpeg = cv2.imencode('.jpg', img)
    latest_frame = jpeg.tobytes()
    return {"message": "Image processed"}

# Trigger feed (ảnh khi té ngã)
@app.get("/trigger_feed")
async def trigger_feed():
    global fall_frame
    if fall_frame:
        return StreamingResponse(BytesIO(fall_frame), media_type="image/jpeg")
    else:
        blank = np.zeros((200, 300, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', blank)
        return StreamingResponse(BytesIO(jpeg.tobytes()), media_type="image/jpeg")

# Nhận metrics
@app.post("/metrics")
async def receive_metrics(data: dict):
    metrics_data["cpu"] = data["cpu"]
    metrics_data["memory"] = data["memory"]
    return {"status": "received"}

@app.get("/get_metrics")
async def get_metrics():
    return metrics_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
