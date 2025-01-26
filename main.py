import cv2
import numpy as np
import time
import tensorflow as tf
import os
from deepface import DeepFace


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # This line is so that only required warnings / logs are shown
emotion_history = []
emotion_counts = {}
max_history_length = 5
last_emotion = None
last_logged_time = time.time()
log_interval = 15  # Log every 15 seconds

# Function to save emotions to a file with timestamps
def save_emotion_to_file(emotion):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open('emotion_history.txt', 'a') as f:
        f.write(f"{timestamp}: {emotion}\n")

def update_emotion_counts(emotion):
    if emotion in emotion_counts:
        emotion_counts[emotion] += 1
    else:
        emotion_counts[emotion] = 1

def draw_emotion_bar_graph(frame):
    max_count = max(emotion_counts.values()) if emotion_counts else 1
    bar_width = 30
    bar_spacing = 10
    start_x = 10
    start_y = 500

    for emotion, count in emotion_counts.items():
        bar_height = int((count / max_count) * 100)
        cv2.rectangle(frame, (start_x, start_y - bar_height), (start_x + bar_width, start_y), (0, 255, 0), -1)
        cv2.putText(frame, emotion, (start_x, start_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        start_x += bar_width + bar_spacing

def start_video():
    global last_emotion, last_logged_time
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
            emotion_history.append(dominant_emotion)
            update_emotion_counts(dominant_emotion)
            current_time = time.time()
            if current_time - last_logged_time >= log_interval:
                save_emotion_to_file(dominant_emotion)
                last_logged_time = current_time

            if len(emotion_history) > max_history_length:
                emotion_history.pop(0)
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.putText(frame, f'Emotion: {dominant_emotion}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            draw_emotion_bar_graph(frame)

        except Exception as e:
            print(f"Error analyzing frame: {e}")

        frame = cv2.resize(frame, (1280, 720))

        # Show the frame with emotion detection
        cv2.imshow("EmotionWise: the emotion detector app", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def display_instructions():
    print("Welcome to the EmotionWise! The emotion detection app!\n")
    print("Instructions:")
    print("- This application will detect emotions from your webcam feed.")
    print("- Press 'q' to quit the video feed.")
    print(f"- The detected emotions will be saved to 'emotion_history.txt' every {log_interval} seconds.\n")

# Main execution
if __name__ == "__main__":
    display_instructions()
    input("Press Enter to launch the camera...")
    start_video()
