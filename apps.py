from flask import Flask, render_template, redirect, url_for, flash, send_file, request, jsonify, Response
import os
import subprocess
import sys
import cv2
import numpy as np
import mediapipe as mp
import base64
import json
import threading
import time
from function import mp_hands, mediapipe_detection, extract_keypoints, actions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Global Model Loading
model = None
model_error = None
model_lock = threading.Lock()
hands_lock = threading.Lock()
hands_processor = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)

def load_keras_model():
    global model, model_error
    try:
        print("⏳ Constructing Sign Language LSTM Model layers...")
        from keras.models import Sequential
        from keras.layers import LSTM, Dense
        
        loaded_model = Sequential()
        loaded_model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
        loaded_model.add(LSTM(128, return_sequences=True, activation='relu'))
        loaded_model.add(LSTM(64, return_sequences=False, activation='relu'))
        loaded_model.add(Dense(64, activation='relu'))
        loaded_model.add(Dense(32, activation='relu'))
        loaded_model.add(Dense(37, activation='softmax'))
        
        print("⏳ Loading pre-trained weights from model.h5...")
        loaded_model.load_weights("model.h5")
        
        # Warmup prediction to speed up first inference
        warmup_seq = np.zeros((1, 30, 63))
        loaded_model.predict(warmup_seq, verbose=0)
        
        with model_lock:
            model = loaded_model
            model_error = None
        print("✅ Sign Language LSTM Model loaded successfully!")
    except Exception as e:
        with model_lock:
            model = None
            model_error = str(e)
        print(f"❌ Error loading model: {e}")

# Load model in a background thread to prevent blocking server start
threading.Thread(target=load_keras_model, daemon=True).start()

# Thread-safe Session for Tracking Real-time translation state
class TranslationSession:
    def __init__(self):
        self.sequence = []
        self.sentence = []           # List of completed words/phrases
        self.accuracy = []
        self.predictions = []
        self.current_word_buffer = []  # In-progress spelled letters
        self.full_string = ""        # Full text (completed words + live word)
        self.latest_label = "No Gesture"
        self.latest_prob = 0.0
        self.last_committed = None
        self.frames_since_commit = 999
        self.frames_no_hand = 0
        self.hand_was_released = True
        self.lock = threading.Lock()
        
    def reset(self):
        with self.lock:
            self.sequence = []
            self.sentence = []
            self.accuracy = []
            self.predictions = []
            self.current_word_buffer = []
            self.full_string = ""
            self.latest_label = "No Gesture"
            self.latest_prob = 0.0
            self.last_committed = None
            self.frames_since_commit = 999
            self.frames_no_hand = 0
            self.hand_was_released = True

    def commit_current_word(self):
        """Explicitly commit letters in current_word_buffer as a complete word."""
        with self.lock:
            if self.current_word_buffer:
                word = ''.join(self.current_word_buffer)
                self.sentence.append(word)
                self.current_word_buffer = []
                self._update_full_string()
                return word
            return None

    def delete_last(self):
        """Backspace: remove last letter from current word, or last word from sentence."""
        with self.lock:
            if self.current_word_buffer:
                self.current_word_buffer.pop()
            elif self.sentence:
                self.sentence.pop()
            self._update_full_string()

    def clear_word_buffer(self):
        """Clear current word in progress."""
        with self.lock:
            self.current_word_buffer = []
            self._update_full_string()

    def _update_full_string(self):
        """Recalculate full_string based on completed words + in-progress word."""
        curr_word = ''.join(self.current_word_buffer)
        all_parts = list(self.sentence)
        if curr_word:
            all_parts.append(curr_word)
        self.full_string = ' '.join(all_parts)

    def process_keypoints(self, keypoints, hand_detected):
        with self.lock:
            if not hand_detected:
                self.frames_no_hand += 1
                self.sequence = []
                self.predictions = []
                self.latest_label = "No Hand"
                self.latest_prob = 0.0
                self.hand_was_released = True
                
                # Auto-commit word after ~18 frames (~0.75s) of resting or releasing hand
                if self.frames_no_hand >= 18 and self.current_word_buffer:
                    word = ''.join(self.current_word_buffer)
                    self.sentence.append(word)
                    self.current_word_buffer = []
                    self._update_full_string()

                return self.latest_label, self.latest_prob, self.full_string, ''.join(self.current_word_buffer)

            # Hand detected
            self.frames_no_hand = 0
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]
            self.frames_since_commit += 1
            
            with model_lock:
                current_model = model
                
            if current_model is None:
                return self.latest_label, self.latest_prob, self.full_string, ''.join(self.current_word_buffer)
                
            try:
                # The existing model was trained from still images repeated across
                # 30 frames, so live inference must use the same input format.
                model_input = np.repeat(keypoints[np.newaxis, :], 30, axis=0)
                res = current_model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
                max_idx = np.argmax(res)
                prob = float(res[max_idx])
                sorted_probs = np.sort(res)
                margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else prob
                predicted_action = actions[max_idx]
                
                self.latest_label = predicted_action
                self.latest_prob = prob
                self.predictions.append(max_idx)
                self.predictions = self.predictions[-15:]
                
                threshold = 0.85
                margin_threshold = 0.15
                stable_window = self.predictions[-8:]
                stable_count = stable_window.count(max_idx)
                is_stable = len(stable_window) == 8 and stable_count >= 7
                cooldown_done = self.frames_since_commit >= 10
                new_or_released = predicted_action != self.last_committed or self.hand_was_released

                if is_stable and cooldown_done and new_or_released and prob >= threshold and margin >= margin_threshold:
                    is_letter = (len(predicted_action) == 1 and predicted_action.isalpha())
                    is_comma = (predicted_action == ',')
                    
                    if is_letter:
                        # Append single letter to current in-progress word buffer
                        self.current_word_buffer.append(predicted_action)
                        self.accuracy.append(f"{prob * 100:.1f}%")
                        self._update_full_string()
                    elif is_comma:
                        # Comma sign acts as space / commit delimiter
                        if self.current_word_buffer:
                            self.sentence.append(''.join(self.current_word_buffer))
                            self.current_word_buffer = []
                        self._update_full_string()
                    else:
                        # Whole-word / multi-word phrase sign (e.g. HELLO, THANK YOU, etc.)
                        if self.current_word_buffer:
                            self.sentence.append(''.join(self.current_word_buffer))
                            self.current_word_buffer = []
                        self.sentence.append(predicted_action)
                        self.accuracy.append(f"{prob * 100:.1f}%")
                        self._update_full_string()

                    self.last_committed = predicted_action
                    self.frames_since_commit = 0
                    self.hand_was_released = False
                    
            except Exception as e:
                print(f"Prediction error: {e}")
                
            return self.latest_label, self.latest_prob, self.full_string, ''.join(self.current_word_buffer)

session_state = TranslationSession()

@app.route('/model_status')
def model_status():
    """Check if model is loaded"""
    with model_lock:
        is_ready = model is not None
        error = model_error
    return jsonify({
        'ready': is_ready,
        'error': error,
        'message': 'Model ready for inference' if is_ready else (error or 'Model is loading, please wait...')
    })

@app.after_request
def set_cache_control(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html', show_animation=True)

@app.route('/home')
def home():
    return render_template('index.html', show_animation=False)

@app.route('/create_gesture')
def create_gesture():
    try:
        subprocess.run([sys.executable, 'create_gesture.py'], cwd=app.root_path, check=True)
        flash('Contact us to create new gesture!', 'success')
    except Exception as e:
        flash(f'Error creating gesture: {e}', 'error')
    return redirect(url_for('home'))

@app.route('/scan_gesture')
def scan_gesture():
    return render_template('scan_gesture.html')

@app.route('/scan_gesture_run', methods=['POST'])
def scan_gesture_run():
    try:
        # Run gesture detection in the background
        subprocess.Popen([sys.executable, 'run_gesture.py'], cwd=app.root_path)
        flash('📷 Webcam opened! Perform sign language gestures. Close the webcam window when done.', 'success')
    except Exception as e:
        flash(f'❌ Error opening webcam: {e}', 'error')
    return redirect(url_for('home'))

# Real-Time base64 frame processing endpoint (WebRTC Mode)
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Check if model is ready
        with model_lock:
            current_model = model
            error = model_error
        
        if current_model is None:
            message = error or 'Model is still loading, please wait...'
            return jsonify({'status': 'error', 'message': message})
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image data'})
            
        # Parse image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image'})
            
        h, w, c = frame.shape
        # Dynamic centered ROI scaled to camera resolution
        roi_size = min(int(min(h, w) * 0.7), 450)
        x1 = (w - roi_size) // 2
        x2 = x1 + roi_size
        y1 = (h - roi_size) // 2
        y2 = y1 + roi_size
        
        cropframe = frame[y1:y2, x1:x2]
        
        with hands_lock:
            image, results = mediapipe_detection(cropframe, hands_processor)
            hand_detected = bool(getattr(results, 'multi_hand_landmarks', None))
            keypoints = extract_keypoints(results)
            
            # Process keypoints with word formation
            label, prob, full_string, current_word = session_state.process_keypoints(keypoints, hand_detected)
            
            # Draw skeletal hands
            from function import draw_styled_landmarks
            draw_styled_landmarks(cropframe, results)
            
            # Put back cropped frame
            frame[y1:y2, x1:x2] = cropframe
            
            # Draw centered ROI rectangle overlay
            cv2.rectangle(frame, (x1, y1), (x2, y2), (97, 218, 251), 2)
            cv2.putText(frame, "Scan Region", (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (97, 218, 251), 1, cv2.LINE_AA)
            
            # Encode processed image back to base64 with high JPEG quality (92)
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ret:
                return jsonify({'status': 'error', 'message': 'Failed to encode processed image'})
                
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            processed_img_url = f"data:image/jpeg;base64,{processed_base64}"
            
            # Fetch Top 5 distribution probabilities
            prob_list = []
            with session_state.lock:
                latest_keypoints = session_state.sequence[-1] if session_state.sequence else None
            
            with model_lock:
                current_model = model
                
            if hand_detected and latest_keypoints is not None and current_model is not None:
                try:
                    model_input = np.repeat(latest_keypoints[np.newaxis, :], 30, axis=0)
                    res_dist = current_model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
                    top_indices = np.argsort(res_dist)[-5:][::-1]
                    prob_list = [{'action': actions[idx], 'prob': float(res_dist[idx])} for idx in top_indices]
                except Exception as e:
                    print(f"Prob distribution error: {e}")
            
            # Write to sentence.txt for fallback compatibility
            with open('sentence.txt', 'w') as file:
                file.write(full_string)
                
            return jsonify({
                'status': 'success',
                'processed_image': processed_img_url,
                'gesture': label,
                'confidence': prob,
                'current_word': current_word,
                'current_letters': list(session_state.current_word_buffer),
                'sentence': ' '.join(session_state.sentence),
                'full_string': full_string,
                'hand_detected': hand_detected,
                'probabilities': prob_list
            })
            
    except Exception as e:
        print(f"Process frame error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

# Real-Time Server-Side OpenCV Capture & Stream fallback (MJPEG Mode)
def generate_mjpeg_frames():
    cap = cv2.VideoCapture(0)
    # Set camera resolution to 1280x720 HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Error: Could not open server webcam")
        # Generate error frame instead of failing silently
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "ERROR: Camera Not Available", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(error_frame, "Please check camera permissions or use Browser Mode", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', error_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ret:
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    # Wait for model to load (max 30 seconds)
    model_ready = False
    wait_count = 0
    while not model_ready and wait_count < 30:
        with model_lock:
            if model is not None:
                model_ready = True
        if not model_ready:
            time.sleep(1)
            wait_count += 1
    
    if not model_ready:
        print("⚠️ Warning: Model did not load in time, proceeding with hand detection only")
        
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65) as hands:
        
        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                print("⚠️ Frame read failed, restarting capture...")
                cap.release()
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                continue
                
            try:
                h, w, c = frame.shape
                # Dynamic centered ROI
                roi_size = min(int(min(h, w) * 0.7), 450)
                x1 = (w - roi_size) // 2
                x2 = x1 + roi_size
                y1 = (h - roi_size) // 2
                y2 = y1 + roi_size
                
                cropframe = frame[y1:y2, x1:x2]
                
                image, results = mediapipe_detection(cropframe, hands)
                hand_detected = bool(getattr(results, 'multi_hand_landmarks', None))
                keypoints = extract_keypoints(results)
                
                label, prob, full_string, current_word = session_state.process_keypoints(keypoints, hand_detected)
                
                # Draw skeletal hands
                from function import draw_styled_landmarks
                draw_styled_landmarks(cropframe, results)
                frame[y1:y2, x1:x2] = cropframe
                
                # Overlay visual HUD on video
                cv2.rectangle(frame, (x1, y1), (x2, y2), (97, 218, 251), 2)
                cv2.putText(frame, f"Scan Region", (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (97, 218, 251), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Gesture: {label} ({prob * 100:.1f}%)", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Sentence: {full_string}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Save fallback sentence text
                with open('sentence.txt', 'w') as file:
                    file.write(full_string)
                    
                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if not ret:
                    continue
                    
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
                frame_count += 1
                time.sleep(0.04)  # ~25 FPS
                
            except Exception as e:
                print(f"❌ Error processing frame: {e}")
                continue
            
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_translation_state')
def get_translation_state():
    with session_state.lock:
        return jsonify({
            'gesture': session_state.latest_label,
            'confidence': session_state.latest_prob,
            'current_word': ''.join(session_state.current_word_buffer),
            'current_letters': list(session_state.current_word_buffer),
            'sentence': ' '.join(session_state.sentence),
            'full_string': session_state.full_string
        })

@app.route('/commit_word', methods=['POST'])
def commit_word():
    committed = session_state.commit_current_word()
    try:
        with open('sentence.txt', 'w') as file:
            file.write(session_state.full_string)
    except Exception as e:
        print(f"Error saving sentence: {e}")
    return jsonify({
        'status': 'success',
        'committed_word': committed,
        'full_string': session_state.full_string,
        'sentence': ' '.join(session_state.sentence),
        'current_word': ''
    })

@app.route('/delete_last', methods=['POST'])
def delete_last():
    session_state.delete_last()
    try:
        with open('sentence.txt', 'w') as file:
            file.write(session_state.full_string)
    except Exception as e:
        print(f"Error saving sentence: {e}")
    return jsonify({
        'status': 'success',
        'full_string': session_state.full_string,
        'sentence': ' '.join(session_state.sentence),
        'current_word': ''.join(session_state.current_word_buffer)
    })

@app.route('/clear_word', methods=['POST'])
def clear_word():
    session_state.clear_word_buffer()
    try:
        with open('sentence.txt', 'w') as file:
            file.write(session_state.full_string)
    except Exception as e:
        print(f"Error saving sentence: {e}")
    return jsonify({
        'status': 'success',
        'full_string': session_state.full_string,
        'sentence': ' '.join(session_state.sentence),
        'current_word': ''
    })

@app.route('/reset_translation', methods=['POST'])
def reset_translation():
    session_state.reset()
    try:
        with open('sentence.txt', 'w') as file:
            file.write('')
    except Exception as e:
        print(f"Error resetting sentence file: {e}")
    return jsonify({'status': 'success', 'message': 'Translation state reset successful'})

@app.route('/show_sentence')
def show_sentence():
    file_path = 'sentence.txt'
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                sentence = file.read()
            return render_template('show_sentence.html', sentence=sentence)
        else:
            flash('Sentence file not found.', 'error')
            return redirect(url_for('home'))
    except Exception as e:
        flash(f'Error showing sentence: {e}', 'error')
        return redirect(url_for('home'))

@app.route('/translate_text', methods=['POST'])
def translate_text():
    """Translate English text to a target Indian language using Google Translate (unofficial)."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        target_lang = data.get('target_lang', 'en')
        
        if not text or target_lang == 'en':
            return jsonify({'status': 'success', 'translated': text, 'lang': target_lang})
        
        # Use Google Translate unofficial API
        import urllib.request
        import urllib.parse
        import json as _json
        
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={urllib.parse.quote(target_lang)}&dt=t&q={urllib.parse.quote(text)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            result = _json.loads(response.read().decode('utf-8'))
            translated = ''.join([seg[0] for seg in result[0] if seg[0]])
            return jsonify({'status': 'success', 'translated': translated, 'lang': target_lang})
            
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({'status': 'fallback', 'translated': data.get('text', ''), 'lang': 'en', 'error': str(e)})

@app.route('/convert_to_audio')
def convert_to_audio():
    try:
        subprocess.run([sys.executable, 'text_to_speech.py'], cwd=app.root_path, check=True)
        flash('Converted to audio successfully!', 'success')
        audio_file = 'output.mp3'
        if os.path.exists(audio_file):
            os.system(f'afplay {audio_file}')  # For MacOS
        else:
            flash('Audio file not found.', 'error')
    except Exception as e:
        flash(f'Error converting to audio: {e}', 'error')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=8000)
