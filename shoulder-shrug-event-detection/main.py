import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def detect_shoulders(frame, prev_positions, shrug_count, movement_history, detected_times, elapsed_time, frame_threshold=15, movement_threshold=5, shrug_cooldown_frames=10, exaggerated_threshold=0.40):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    shrug_detected = False

    if not hasattr(detect_shoulders, "shrug_start_time"):
        detect_shoulders.shrug_start_time = None
    if not hasattr(detect_shoulders, "shrug_cooldown"):
        detect_shoulders.shrug_cooldown = 0
    if not hasattr(detect_shoulders, "movement_state"):
        detect_shoulders.movement_state = "neutral"

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_left = gray[max(0, y + h): min(y + h + int(h * 0.5), gray.shape[0]),
                        max(0, x - int(w * 0.3)): max(0, x)]
        roi_right = gray[max(0, y + h): min(y + h + int(h * 0.5), gray.shape[0]),
                         min(x + w, gray.shape[1]): min(x + w + int(w * 0.3), gray.shape[1])]

        if roi_left.size == 0 or roi_right.size == 0:
            continue

        edges_left = cv2.Canny(roi_left, 50, 150)
        edges_right = cv2.Canny(roi_right, 50, 150)

        def process_edges(edges, x_offset, y_offset):
            lines = []
            for y, x in zip(*np.where(edges)):
                lines.append((x + x_offset, y + y_offset))
            return lines

        lines_left = process_edges(edges_left, x - int(w * 0.3), y + h)
        lines_right = process_edges(edges_right, x + w, y + h)

        left_y = None
        right_y = None

        if lines_left:
            x_vals, y_vals = zip(*lines_left)
            left_start = (max(x_vals), min(y_vals))
            left_end = (min(x_vals), max(y_vals))
            left_y = min(y_vals)

        if lines_right:
            x_vals, y_vals = zip(*lines_right)
            right_start = (min(x_vals), min(y_vals))
            right_end = (max(x_vals), max(y_vals))
            right_y = min(y_vals)

        if left_y is not None and right_y is not None:
            movement_history['left'].append(left_y)
            movement_history['right'].append(right_y)

            if len(movement_history['left']) > frame_threshold:
                smoothed_left = gaussian_filter(np.array(movement_history['left']), sigma=1)
                movement_history['left'] = smoothed_left.tolist()
                movement_history['left'].pop(0)

            if len(movement_history['right']) > frame_threshold:
                smoothed_right = gaussian_filter(np.array(movement_history['right']), sigma=1)
                movement_history['right'] = smoothed_right.tolist()
                movement_history['right'].pop(0)

            if len(movement_history['left']) == frame_threshold and len(movement_history['right']) == frame_threshold:
                left_diff = max(movement_history['left']) - min(movement_history['left'])
                right_diff = max(movement_history['right']) - min(movement_history['right'])

                if left_diff > movement_threshold and right_diff > movement_threshold:
                    if detect_shoulders.movement_state == "neutral" and movement_history['left'][-1] < movement_history['left'][0]:
                        detect_shoulders.movement_state = "up"
                        detect_shoulders.shrug_start_time = elapsed_time 
                    elif detect_shoulders.movement_state == "up" and movement_history['left'][-1] > movement_history['left'][0]:
                        detect_shoulders.movement_state = "down"
                        if detect_shoulders.shrug_cooldown == 0:
                            shrug_duration = elapsed_time - detect_shoulders.shrug_start_time  # Calculate shrug duration
                            shrug_type = "exaggerated" if shrug_duration > exaggerated_threshold else "normal"
                            print(f"Shrug detected at {elapsed_time:.2f} seconds ({shrug_type}). Duration: {shrug_duration:.2f} seconds.")
                            shrug_count += 1
                            shrug_detected = True
                            detected_times[shrug_type].append(elapsed_time)

                            movement_history['left'] = movement_history['left'][-5:]
                            movement_history['right'] = movement_history['right'][-5:]

                            detect_shoulders.shrug_cooldown = shrug_cooldown_frames
                            detect_shoulders.movement_state = "neutral"

        line_color = (0, 0, 255) if shrug_detected else (0, 255, 0)

        if lines_left:
            cv2.line(frame, left_start, left_end, line_color, 2)
        if lines_right:
            cv2.line(frame, right_start, right_end, line_color, 2)

        break

    if detect_shoulders.shrug_cooldown > 0:
        detect_shoulders.shrug_cooldown -= 1

    return frame, shrug_count


def display_videos(video_folder, num_videos, ground_truth, time_threshold=0.5):
    detected_times_per_video = []

    for i in range(1, num_videos + 1):
        video_path = f"{video_folder}\\video{i}.mp4"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}.")
            detected_times_per_video.append({})
            continue

        print(f"\nDisplaying video: {video_path}...")

        shrug_count = 0
        prev_positions = [None, None]
        movement_history = {'left': [], 'right': []}
        detected_times = {"normal": [], "exaggerated": []}  # Changed to a dictionary

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Video {video_path} completed.")
                break

            frame_count += 1
            elapsed_time = frame_count / fps

            frame, shrug_count = detect_shoulders(
                frame, prev_positions, shrug_count, movement_history, detected_times, elapsed_time
            )

            cv2.putText(frame, f"Time: {elapsed_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Video Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting early on user request.")
                break

        cap.release()

        print(f"Total shrugs detected in video {i}: {shrug_count}")
        print(f"Detected shrug times for video {i}: {detected_times}")
        detected_times_per_video.append(detected_times)

        print(f"\nProcessing ROC for Video {i}...")
        fpr, tpr, roc_auc = calculate_roc_curve(ground_truth[i - 1], detected_times["normal"] + detected_times["exaggerated"], time_threshold)

        print(f"Video {i} ROC Metrics:")
        print(f"False Positive Rates: {fpr}")
        print(f"True Positive Rates: {tpr}")
        print(f"Area Under Curve (AUC): {roc_auc:.2f}")

        plot_roc_curve(fpr, tpr, roc_auc, video_id=i)

    print("All videos processed and displayed successfully.")
    cv2.destroyAllWindows()
    return detected_times_per_video

def calculate_roc_curve(ground_truth, detected_times, max_time_threshold=0.5):
    if not ground_truth and not detected_times:
        print("No ground truth or detected times available for ROC calculation.")
        return [0], [0], 0.0

    tpr_list = []
    fpr_list = []

    for time_threshold in np.linspace(0.1, max_time_threshold, 50):  
        tp, fp, fn = 0, 0, 0

        for gt_time in ground_truth:
            if any(abs(gt_time - det_time) <= time_threshold for det_time in detected_times):
                tp += 1
            else:
                fn += 1

        for det_time in detected_times:
            if not any(abs(gt_time - det_time) <= time_threshold for gt_time in ground_truth):
                fp += 1

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + len(ground_truth)) if (fp + len(ground_truth)) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr_list = [0] + tpr_list + [1]
    fpr_list = [0] + fpr_list + [1]

    fpr_list = sorted(fpr_list)
    tpr_list = sorted(tpr_list)

    unique_fpr = []
    unique_tpr = []
    for fpr, tpr in zip(fpr_list, tpr_list):
        if unique_fpr and fpr == unique_fpr[-1]:
            unique_tpr[-1] = max(unique_tpr[-1], tpr)
        else:
            unique_fpr.append(fpr)
            unique_tpr.append(tpr)

    roc_auc = auc(unique_fpr, unique_tpr)

    return unique_fpr, unique_tpr, roc_auc


def plot_roc_curve(fpr, tpr, roc_auc, video_id):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess")
    plt.scatter(fpr, tpr, color='red', label="ROC Points") 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Video {video_id}')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    video_folder = '..\\shoulder-shrug-event-detection\\sample-videos'
    num_videos = 3

    ground_truth = [
        [4.0, 15.0, 25.0, 35.0, 43.0, 50.0, 52.0],  # Video 1
        [14.0, 16.0, 21.0, 29.0, 36.0, 44.0],  # Video 2
        [5.0, 14.0, 24.0, 29.0, 34.0, 49.0, 53.0, 56.0]   # Video 3
    ]

    detected_times = display_videos(video_folder, num_videos, ground_truth)
