import argparse
import os
import time

import cv2
import numpy as np

from function import DATA_PATH, actions, draw_styled_landmarks, extract_keypoints, mediapipe_detection, mp_hands, sequence_length


def next_sequence_id(action):
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)
    existing = [int(name) for name in os.listdir(action_path) if name.isdigit()]
    return max(existing, default=-1) + 1


def parse_args():
    parser = argparse.ArgumentParser(description="Collect live MediaPipe landmark sequences for better gesture accuracy.")
    parser.add_argument("action", choices=list(actions), help="Gesture label to collect.")
    parser.add_argument("--sequences", type=int, default=30, help="Number of 30-frame samples to collect.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--warmup", type=float, default=1.0, help="Seconds to wait before each sample.")
    return parser.parse_args()


def main():
    args = parse_args()
    start_id = next_sequence_id(args.action)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print(f"Collecting {args.sequences} sequences for: {args.action}")
    print("Keep your hand inside the rectangle. Press q to stop early.")

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    ) as hands:
        for sequence_offset in range(args.sequences):
            sequence_id = start_id + sequence_offset
            sequence_path = os.path.join(DATA_PATH, args.action, str(sequence_id))
            os.makedirs(sequence_path, exist_ok=True)

            wait_until = time.time() + args.warmup
            while time.time() < wait_until:
                ret, frame = cap.read()
                if not ret:
                    continue
                cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
                cv2.putText(frame, f"Ready: {args.action} #{sequence_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Collect Live Sequences", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            saved_frames = 0
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue

                cropframe = frame[40:400, 0:300]
                image, results = mediapipe_detection(cropframe, hands)
                hand_detected = bool(getattr(results, "multi_hand_landmarks", None))
                keypoints = extract_keypoints(results) if hand_detected else np.zeros(21 * 3)

                np.save(os.path.join(sequence_path, str(frame_num)), keypoints)
                saved_frames += 1

                draw_styled_landmarks(cropframe, results)
                frame[40:400, 0:300] = cropframe
                cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
                cv2.putText(frame, f"{args.action} #{sequence_id} frame {frame_num + 1}/{sequence_length}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Collect Live Sequences", frame)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print(f"Saved sequence {sequence_id} with {saved_frames} frames.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
