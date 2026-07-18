from flask import Flask, Response
import cv2
import time

app = Flask(__name__)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 在 Windows 端先要求相機輸出格式
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# 檢查實際相機設定
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print("Camera opened:", cap.isOpened())
print("Actual width:", actual_width)
print("Actual height:", actual_height)
print("Actual fps:", actual_fps)


def gen_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            time.sleep(0.01)
            continue

        # 保證送出的影像一定是 640x480
        frame = cv2.resize(frame, (640, 480))

        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            print("Failed to encode frame")
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg.tobytes() +
            b"\r\n"
        )

        # 控制大約 30 FPS，避免網路串流過快
        time.sleep(1 / 30)


@app.route("/video")
def video():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


app.run(host="0.0.0.0", port=5000, threaded=True)