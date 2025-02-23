import cv2
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
MAX_FRAMES = 30  # Number of frames used for classification

# Load label encoder
LABELS = ["book", "drink", "computer_bk"]  # Same classes as in training

def extract_landmarks(video_path, output_csv_path):
    """
    Extracts normalized (x, y) landmarks from a video and saves them to a CSV file.
    """
    output_csv_path = os.path.join(output_csv_path, "landmarks.csv")
    
    print(f"Start extract_landmarks {video_path}, {output_csv_path}....")            
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video at {video_path}")
        return None

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands, \
            mp_pose.Pose(static_image_mode=False) as pose:
        
        all_frames_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb_frame)
            pose_results = pose.process(rgb_frame)

            frame_data = [0.0] * FEATURE_SIZE  # Initialize with zeros

            #print(f"Start processing hands ....")

            # Process hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    base_idx = 0 if hand_results.multi_handedness[hand_idx].classification[0].label == "Left" else LEFT_HAND_LANDMARKS * 2
                    for i, lm in enumerate(hand_landmarks.landmark):
                        frame_data[base_idx + i * 2] = lm.x
                        frame_data[base_idx + i * 2 + 1] = lm.y

            #print(f"Start processing pose ....")
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
    
    print(f"Convert to numpy array ....")

    # Ensure MAX_FRAMES length
    if len(all_frames_data) < MAX_FRAMES:
        padding = np.zeros((MAX_FRAMES - len(all_frames_data), FEATURE_SIZE))
        all_frames_data = np.vstack((all_frames_data, padding))
    else:
        step = max(1, len(all_frames_data) // MAX_FRAMES)
        all_frames_data = all_frames_data[::step][:MAX_FRAMES]

    # Save to CSV
    df = pd.DataFrame(all_frames_data)
    df.to_csv(output_csv_path, index=False, header=False)
    print(f"✅ Saved landmarks to: {output_csv_path}")

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
    Processes video to generate joint visualization and classifies the sign.
    """
    print(f"Start generate_joints ....")

    cap = cv2.VideoCapture(input_video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    iFrame = 0
    print(f"Start generate_joints 2....")    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands, \
            mp_pose.Pose(static_image_mode=False) as pose:

        #print(f"Start generate_joints 3....")        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            black_frame = np.zeros((height, width, 3), dtype=np.uint8)  # Black background
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            #print(f"Start generate_joints 4....")
            iFrame += 1
            #print(f"Procesing frame {iFrame}")            

            hand_results = hands.process(rgb_frame)
            pose_results = pose.process(rgb_frame)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        black_frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    black_frame, 
                    pose_results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style()
                )

            out.write(black_frame)

    cap.release()
    out.release()
    
    print(f"Finished processing video ")            
    predicted_sign = classify_video(input_video_path, outputs_dir)
    print(f"✅ Predicted sign: {predicted_sign}")
    return predicted_sign