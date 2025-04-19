from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import threading

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
    global latest_frame
    content = await file.read()
    latest_frame = content
    return {"message": "Image received successfully"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)