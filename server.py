from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import threading
import time
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
import collections # Thêm thư viện collections

app = FastAPI()

# Cung cấp thư mục static
app.mount("/custom", StaticFiles(directory="custom"), name="custom")

# Load pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Tăng độ tin cậy tối thiểu để lọc bỏ các phát hiện kém
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Shared states
latest_frame = None
fall_frame = None
last_fall_time = 0
metrics_data = {"cpu": 0, "memory": 0}

# --- Biến trạng thái cho phát hiện té ngã ---
# Lưu trữ lịch sử vị trí Y của mũi và hông trung bình trong vài khung hình gần nhất
pose_history = collections.deque(maxlen=5) # Lưu 5 khung hình gần nhất
is_falling = False # Cờ trạng thái đang té ngã
fall_cooldown = 10 # Giây - Thời gian hồi để tránh trigger liên tục

# --- Hằng số cấu hình cho phát hiện té ngã ---
# Ngưỡng thay đổi vị trí Y của mũi giữa các khung hình để được xem là chuyển động nhanh
# Cần điều chỉnh giá trị này dựa trên thực tế!
VERTICAL_VELOCITY_THRESHOLD = 0.08 # Tọa độ chuẩn hóa (0.0 - 1.0)

# Ngưỡng vị trí cuối cùng: Mũi phải thấp hơn hông bao nhiêu để xác nhận ngã
# Giá trị dương vì Y tăng xuống dưới
FINAL_POSITION_THRESHOLD_OFFSET = 0.03

# Ngưỡng visibility của landmark để sử dụng
VISIBILITY_THRESHOLD = 0.6

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

# === HÀM PHÁT HIỆN TÉ NGÃ ĐƯỢC CẢI TIẾN ===
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global latest_frame, fall_frame, last_fall_time, pose_history, is_falling

    start_time = time.time() # Đo thời gian xử lý (tùy chọn)

    content = await file.read()
    np_img = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        print("Lỗi: Không thể decode ảnh")
        return {"message": "Image decode error"}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    detected_fall_this_frame = False
    current_pose_data = None

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

            # Lưu trữ dữ liệu hiện tại
            current_pose_data = {"nose_y": current_nose_y, "avg_hip_y": avg_hip_y, "time": time.time()}
            pose_history.append(current_pose_data)

            # --- Logic phát hiện té ngã mới ---
            # Cần ít nhất 2 điểm dữ liệu trong lịch sử để tính vận tốc
            if len(pose_history) >= 2:
                prev_pose_data = pose_history[-2] # Dữ liệu của khung hình ngay trước đó

                # Tính toán sự thay đổi vị trí Y của mũi (vận tốc dọc gần đúng)
                delta_nose_y = current_nose_y - prev_pose_data["nose_y"]
                time_diff = current_pose_data["time"] - prev_pose_data["time"]

                # Tránh chia cho 0 nếu các khung hình quá gần nhau
                vertical_velocity = delta_nose_y / time_diff if time_diff > 0.001 else 0

                # Điều kiện 1: Vận tốc rơi đủ lớn (thay đổi Y dương đáng kể)
                is_moving_down_fast = vertical_velocity > VERTICAL_VELOCITY_THRESHOLD

                # Điều kiện 2: Vị trí cuối cùng thấp (mũi thấp hơn hông)
                is_in_low_position = current_nose_y > (avg_hip_y + FINAL_POSITION_THRESHOLD_OFFSET)

                # Debug print
                print(f"TimeDiff: {time_diff:.3f}, DeltaNoseY: {delta_nose_y:.3f}, Velocity: {vertical_velocity:.3f}, NoseY: {current_nose_y:.3f}, AvgHipY: {avg_hip_y:.3f}, Fast: {is_moving_down_fast}, Low: {is_in_low_position}")

                # Kiểm tra phát hiện té ngã
                current_time = time.time()
                if is_moving_down_fast and is_in_low_position:
                    # Kiểm tra cooldown
                    if current_time - last_fall_time > fall_cooldown:
                        detected_fall_this_frame = True
                        is_falling = True # Bắt đầu trạng thái ngã
                        last_fall_time = current_time # Cập nhật thời điểm ngã cuối cùng
                        print(f"!!! PHÁT HIỆN TÉ NGÃ !!! Vận tốc: {vertical_velocity:.3f}")

                        # Lưu ảnh tại thời điểm phát hiện ngã
                        _, jpeg_fall = cv2.imencode('.jpg', img)
                        fall_frame = jpeg_fall.tobytes()
                    else:
                        print(f"Phát hiện chuyển động giống ngã, nhưng đang trong thời gian cooldown.")
                # else:
                    # Cân nhắc reset cờ is_falling nếu người dùng đứng dậy?
                    # Ví dụ: nếu mũi lại cao hơn hông một cách ổn định
                    # if current_nose_y < avg_hip_y - 0.1: # Nếu mũi cao hơn hông đáng kể
                    #     is_falling = False # Có thể đã đứng dậy
        else:
            print("Cảnh báo: Không đủ visibility của landmarks quan trọng.")
            # Xóa lịch sử nếu không thấy rõ người trong một thời gian? (Tùy chọn)
            # pose_history.clear()


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
        # Trả về ảnh đã lưu trữ khi ngã
        return StreamingResponse(BytesIO(fall_frame), media_type="image/jpeg")
    else:
        # Trả về ảnh trống nếu chưa có cú ngã nào được ghi lại
        blank = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.putText(blank, "No Fall Detected", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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