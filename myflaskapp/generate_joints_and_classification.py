import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
import threading


# Load MediaPipe Hand and Pose Landmarker
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

MODEL_PATH = "/home/mikagrin/ASL_project/myflaskapp/weights/asl_lstm_single_layer.h5"

# Define constants
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
FEATURES_PER_FRAME = (LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS + POSE_LANDMARKS) * 2  # X, Y for each landmark
MAX_FRAMES = 15  # Number of frames used for classification

# Load label encoder
LABELS = ['book', 'computer_bk', 'drink', 'i', 'science', 'study']  # Same classes as in training

# === הורדת מודלים ===
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"✅ LABELS: '{LABELS}'")

BASE_DIR = "/home/mikagrin/ASL_project/myflaskapp/"
print(f"✅ BASE_DIR: '{BASE_DIR}'")

POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker.task")
print(f"✅ POSE_MODEL_PATH: '{POSE_MODEL_PATH}'")

HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
print(f"✅ HAND_MODEL_PATH: '{HAND_MODEL_PATH}'")


def download_model(url, path):
    if not os.path.exists(path):
        print(f"📥 Downloading {path} ...")
        urllib.request.urlretrieve(url, path)
        print(f"✅ Downloaded {path}")

#download_model(
#    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
#    POSE_MODEL_PATH
#)

#download_model(
#    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
#    HAND_MODEL_PATH
#)

# Global variables
pose_landmarker = None
pose_lock = threading.Lock()


model = None
hand_landmarker = None
pose_landmarker = None




#####################################################################################################
#   extract_landmark_matrix_full_video
#####################################################################################################

def extract_landmark_matrix_full_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ total_frames: '{total_frames}'")

    n_frames = min(total_frames, MAX_FRAMES)
    print(f"✅ n_frames: '{n_frames}'")

    selected_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    print(f"✅ selected_indices: '{selected_indices}'")

    frames = []
    current_frame = 0
    selected_set = set(selected_indices)

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    else:
        out = None

    print(f"✅ OUT VIDEO 'writer: '{out}'")


    ####### Init Models ################

    base_options_hand = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    print(f"✅ base_options_hand: '{base_options_hand}'")
    options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=2)
    print(f"✅ options_hand: '{options_hand}'")
    hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)
    print(f"✅ hand_landmarker: '{hand_landmarker}'")

    base_options_pose = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options_pose = vision.PoseLandmarkerOptions(base_options=base_options_pose)
    pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)


    ###################################

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(frames) >= n_frames:
            break

        if current_frame in selected_set:
            print(f"✅ current_frame: '{current_frame}'")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"✅ rgb: '{rgb}'")

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


            print(f"✅ mp_image: '{mp_image}'")

            hand_result = hand_landmarker.detect(mp_image)

            print(f"✅ hand_result: '{hand_result}'")



            with pose_lock:
                pose_result = pose_landmarker.detect(mp_image)
                print(f"✅ pose_result: '{pose_result}'")



            #annotated_image = mp_image.numpy_view()
            #if hand_result.hand_landmarks:
            #    annotated_image = draw_landmarks_on_image(annotated_image, hand_result)
            #if pose_result.pose_landmarks:
            #   annotated_image = draw_pose_landmarks_on_image(annotated_image, pose_result)

            #if out:
            #    bgr_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            #    out.write(bgr_frame)

            # Collect landmark data for classification
            frame_data = [0.0] * FEATURES_PER_FRAME

            if hand_result.hand_landmarks:
                for idx, hand in enumerate(hand_result.hand_landmarks):
                    base = 0 if hand_result.handedness[idx][0].category_name == "Left" else 21 * 2
                    for lm_idx, lm in enumerate(hand):
                        frame_data[base + lm_idx * 2] = lm.x
                        frame_data[base + lm_idx * 2 + 1] = lm.y

            if pose_result.pose_landmarks:
                base = (21 + 21) * 2
                for lm_idx, lm in enumerate(pose_result.pose_landmarks[0]):
                    frame_data[base + lm_idx * 2] = lm.x
                    frame_data[base + lm_idx * 2 + 1] = lm.y

            frames.append(frame_data)

        elif out:
            out.write(frame)

        current_frame += 1

    cap.release()
    if out:
        out.release()

    while len(frames) < n_frames:
        frames.append([0.0] * FEATURES_PER_FRAME)

    frames = np.array(frames[:n_frames])
    return frames


#####################################################################################################
#   classify_video
#####################################################################################################

def classify_video(video_path, outputs_dir):
    """
    Classifies a video based on extracted landmarks using the trained RNN model.
    """
    print(f"Start classify video {video_path} ....")

    model = tf.keras.models.load_model(MODEL_PATH)

    sequence = extract_landmark_matrix_full_video(video_path, outputs_dir)
    print(f"🟡 Processing video: {video_path}")
    if sequence is None:
        return "Error: Could not extract landmarks."

    sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
    print(f"🔢 Input shape to model: {sequence.shape}", flush=True)

    print(f"Start predict ....")
    predictions = model.predict(sequence)
    predicted_label = LABELS[np.argmax(predictions)]

    #predicted_label = "book"
    return predicted_label


#####################################################################################################
#   generate_joints
#####################################################################################################

def generate_joints(input_video_path, output_video_path, outputs_dir):
    """
    This function only classifies the input video by extracting landmarks
    and predicting the sign using the trained model.
    It does not modify or write any new video files.
    """
    print(f"🎬 [generate_joints] Starting classification for: {input_video_path}", flush=True)

    try:
        # פשוט מעבירים את הווידאו לפונקציית הסיווג
        predicted_sign = classify_video(input_video_path, outputs_dir)
        print(f"✅ [generate_joints] Predicted sign: {predicted_sign}", flush=True)
        return predicted_sign

    except Exception as e:
        print(f"❌ [generate_joints] Exception: {str(e)}", flush=True)
        return "Error: failed to classify video"
