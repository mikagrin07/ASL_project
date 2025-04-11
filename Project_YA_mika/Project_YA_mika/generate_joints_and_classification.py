import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
import os

# Load MediaPipe Hand and Pose Landmarker
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load trained RNN model
MODEL_PATH = "C:/Users/User/Downloads/MS-ASL signlanguage Project/ASL_project/asl_lstm_fixed.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define constants
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
FEATURE_SIZE = (LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS + POSE_LANDMARKS) * 2  # X, Y for each landmark
MAX_FRAMES = 15  # Number of frames used for classification

# Load label encoder
LABELS = ["book", "drink", "computer_bk", "study", "i", "science"]  # Same classes as in training

def extract_landmarks(video_path, output_csv_path):
    """
    Extracts normalized (x, y) landmarks from a video and saves them to a CSV file.
    Processes only `MAX_FRAMES` evenly spaced frames.
    """
    output_csv_path = os.path.join(output_csv_path, "landmarks.csv")
    
    print(f"Start extract_landmarks {video_path}, {output_csv_path}....")            
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video at {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
    frame_indices = np.linspace(0, total_frames - 1, num=MAX_FRAMES, dtype=int)  # Select evenly spaced frames

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands, \
            mp_pose.Pose(static_image_mode=False) as pose:
        
        all_frames_data = []

        for frame_idx in frame_indices:
            #print(f"Processinf frame {frame_idx} ....")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Jump to the selected frame
            ret, frame = cap.read()
            if not ret:
                break  # Stop if we can't read the frame

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb_frame)
            pose_results = pose.process(rgb_frame)

            frame_data = [0.0] * FEATURE_SIZE  # Initialize with zeros

            # Process hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    base_idx = 0 if hand_results.multi_handedness[hand_idx].classification[0].label == "Left" else LEFT_HAND_LANDMARKS * 2
                    for i, lm in enumerate(hand_landmarks.landmark):
                        frame_data[base_idx + i * 2] = lm.x
                        frame_data[base_idx + i * 2 + 1] = lm.y

            # Process pose landmarks
            if pose_results.pose_landmarks:
                base_idx = (LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS) * 2
                for i, lm in enumerate(pose_results.pose_landmarks.landmark):
                    frame_data[base_idx + i * 2] = lm.x
                    frame_data[base_idx + i * 2 + 1] = lm.y

            all_frames_data.append(frame_data)

    cap.release()

    # Convert to NumPy array
    all_frames_data = np.array(all_frames_data)
    
    print(f"Convert to numpy array {time.perf_counter()} ....")

    # Ensure that the output is always of length MAX_FRAMES
    if len(all_frames_data) < MAX_FRAMES:
        padding = np.zeros((MAX_FRAMES - len(all_frames_data), FEATURE_SIZE))
        all_frames_data = np.vstack((all_frames_data, padding))

    # Save to CSV (Uncomment if needed)
    # df = pd.DataFrame(all_frames_data)
    # df.to_csv(output_csv_path, index=False, header=False)
    print(f"✅ {time.perf_counter()} Saved landmarks to: {output_csv_path}")

    return all_frames_data


def classify_video(video_path, outputs_dir):
    """
    Classifies a video based on extracted landmarks using the trained RNN model.
    """
    print(f"Start classify video {video_path} ....")   
    
    sequence = extract_landmarks(video_path, outputs_dir)
    if sequence is None:
        return "Error: Could not extract landmarks."

    sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
    
    print(f"Start predict ....")            
    predictions = model.predict(sequence)
    predicted_label = LABELS[np.argmax(predictions)]

    return predicted_label

def generate_joints(input_video_path, output_video_path, outputs_dir):
    """
    Processes video to generate joint visualization using MAX_FRAMES evenly spaced frames.
    """
    print(f"Start generate_joints ....")

    cap = cv2.VideoCapture(input_video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Select MAX_FRAMES evenly spaced indices
    frame_indices = np.linspace(0, total_frames - 1, num=MAX_FRAMES, dtype=int)

    print(f"Start processing video with {MAX_FRAMES} evenly spaced frames...")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands, \
            mp_pose.Pose(static_image_mode=False) as pose:

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Jump to the selected frame
            ret, frame = cap.read()
            if not ret:
                break

            black_frame = np.zeros((height, width, 3), dtype=np.uint8)  # Black background
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            print(f"Processing frame {frame_idx}...")

            hand_results = hands.process(rgb_frame)
            pose_results = pose.process(rgb_frame)

            # Draw hand landmarks
            #if hand_results.multi_hand_landmarks:
            #    for hand_landmarks in hand_results.multi_hand_landmarks:
            #        mp_drawing.draw_landmarks(
            #            black_frame, 
            #            hand_landmarks, 
            #            mp_hands.HAND_CONNECTIONS
            #        )

            # Draw pose landmarks
            #if pose_results.pose_landmarks:
            #    mp_drawing.draw_landmarks(
            #        black_frame, 
            #        pose_results.pose_landmarks, 
            #        mp_pose.POSE_CONNECTIONS
            #    )

            # Write the processed frame to output video
            out.write(black_frame)

    cap.release()
    out.release()

    print(f"✅ Finished processing video with {MAX_FRAMES} frames")            

    # Classify the processed video
    predicted_sign = classify_video(input_video_path, outputs_dir)
    print(f"✅ Predicted sign: {predicted_sign}")
    return predicted_sign