#!/usr/bin/env python3
import subprocess
import sys
import os

# Change to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Starting gesture detection...")
    from function import *
    from keras.utils import to_categorical
    from keras.models import model_from_json
    from keras.layers import LSTM, Dense
    from keras.callbacks import TensorBoard
    import numpy as np
    import mediapipe as mp
    import cv2

    print("✅ All imports successful!")
    
    # Load the model
    json_file = open("model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("model.h5")
    print("✅ Model loaded!")

    colors = [(245, 117, 16) for _ in range(20)]
    
    def prob_viz(res, actions, input_frame, colors, threshold):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return output_frame

    # Detection variables
    sequence = []
    sentence = []
    accuracy = []
    predictions = []
    threshold = 0.8 
    full_string = ""
    file_path = "sentence.txt"

    print("📷 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam!")
        sys.exit(1)
    
    print("✅ Webcam opened successfully!")

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            cropframe = frame[40:400, 0:300]
            frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
            image, results = mediapipe_detection(cropframe, hands)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try: 
                if len(sequence) > 0:
                    model_input = np.repeat(keypoints[np.newaxis, :], 30, axis=0)
                    res = model.predict(np.expand_dims(model_input, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                    if np.unique(predictions[-10:])[0] == np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    accuracy.append(str(res[np.argmax(res)] * 100))
                                    full_string += actions[np.argmax(res)]
                            else:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)] * 100))
                                full_string += actions[np.argmax(res)]

                    if len(sentence) > 1: 
                        sentence = sentence[-1:]
                        accuracy = accuracy[-1:]

            except Exception as e:
                print(f"Detection error: {e}")

            # Write to file
            with open(file_path, 'w') as file:
                file.write(full_string)

            # Display
            cv2.imshow('Sign Language Detection', frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Gesture detection closed")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
