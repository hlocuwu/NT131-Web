from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
# import threading # Không còn sử dụng trong code này
import time
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
# import collections # Không còn sử dụng collections

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
    # Loại bỏ pose_history và is_falling khỏi global nếu không dùng ở đâu khác
    global latest_frame, fall_frame, last_fall_time

    start_time = time.time() # Đo thời gian xử lý (tùy chọn)

    content = await file.read()
    np_img = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        print("Lỗi: Không thể decode ảnh")
        return {"message": "Image decode error"}

    # Giảm kích thước ảnh để tăng tốc độ xử lý (tùy chọn, có thể ảnh hưởng độ chính xác)
    # height, width = img.shape[:2]
    # if width > 640:
    #    scale = 640 / width
    #    img = cv2.resize(img, (640, int(height * scale)), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    detected_fall_this_frame = False
    # current_pose_data không còn cần thiết

    if results.pose_landmarks:
        # Vẽ landmarks lên ảnh gốc (màu BGR)
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        landmarks = results.pose_landmarks.landmark

        # Lấy các landmark cần thiết và kiểm tra visibility
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Chỉ xử lý nếu các landmark quan trọng được nhìn thấy rõ ràng
        if (nose.visibility > VISIBILITY_THRESHOLD and
            left_hip.visibility > VISIBILITY_THRESHOLD and
            right_hip.visibility > VISIBILITY_THRESHOLD):

            avg_hip_y = (left_hip.y + right_hip.y) / 2
            current_nose_y = nose.y

            # --- Logic phát hiện té ngã ĐƠN GIẢN ---
            # Chỉ cần kiểm tra mũi có thấp hơn hông không (Y lớn hơn)
            is_in_low_position = current_nose_y > avg_hip_y

            # Debug print
            print(f"NoseY: {current_nose_y:.3f}, AvgHipY: {avg_hip_y:.3f}, Low: {is_in_low_position}")

            # Kiểm tra phát hiện "ngã" (nằm xuống)
            current_time = time.time()
            if is_in_low_position:
                # Kiểm tra cooldown
                if current_time - last_fall_time > fall_cooldown:
                    detected_fall_this_frame = True
                    # is_falling = True # Không cần thiết nữa
                    last_fall_time = current_time # Cập nhật thời điểm phát hiện cuối cùng
                    print(f"!!! PHÁT HIỆN NẰM XUỐNG (Simplified Trigger) !!!")

                    # Lưu ảnh tại thời điểm phát hiện
                    _, jpeg_fall = cv2.imencode('.jpg', img)
                    fall_frame = jpeg_fall.tobytes()
                else:
                    # Vẫn đang nằm và trong thời gian cooldown
                    # print(f"Đang nằm, trong cooldown.") # Có thể bỏ print này cho đỡ nhiễu
                    pass # Không làm gì nếu đang cooldown
        else:
            print("Cảnh báo: Không đủ visibility của landmarks quan trọng.")


    # --- Cập nhật frame và trả về ---
    # Luôn cập nhật latest_frame để hiển thị stream chính
    _, jpeg = cv2.imencode('.jpg', img)
    latest_frame = jpeg.tobytes()

    processing_time = time.time() - start_time
    # print(f"Thời gian xử lý frame: {processing_time:.4f} giây") # Debug thời gian

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

if __name__ == "__main__":
    import uvicorn
    # Chạy với reload=False khi deploy hoặc nếu không cần tự động tải lại khi sửa code
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False) # Tắt reload để tránh reset state khi có request