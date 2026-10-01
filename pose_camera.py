import sys

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = "models/pose_landmarker_lite.task"
VIDEO_PATH = "./input/input2.mp4"

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"เปิดไฟล์วิดีโอไม่ได้: {VIDEO_PATH}")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(image, timestamp_ms)

        for landmarks in result.pose_landmarks:
            vision.drawing_utils.draw_landmarks(
                frame, landmarks, vision.PoseLandmarksConnections.POSE_LANDMARKS
            )

        cv2.imshow("Pose (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
