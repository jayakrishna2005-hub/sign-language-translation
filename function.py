#import dependency
import cv2
import numpy as np
import os

# ─────────────────────────────────────────────────────────
#  MediaPipe setup – supports both old (mp.solutions) and
#  new (mp.tasks / HandLandmarker) APIs automatically.
# ─────────────────────────────────────────────────────────

USE_MOCK_MEDIAPIPE = False
_HAND_LANDMARKER_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# ── Attempt 1: legacy mp.solutions (mediapipe < 0.10) ────
try:
    import mediapipe as mp
    _solutions = getattr(mp, 'solutions', None)
    if _solutions is not None and hasattr(_solutions, 'hands'):
        mp_drawing = _solutions.drawing_utils
        mp_drawing_styles = _solutions.drawing_styles
        mp_hands = _solutions.hands
        print(f"✅ MediaPipe loaded via mp.solutions (v{mp.__version__})")
        _USE_TASKS_API = False
    else:
        raise ImportError("mp.solutions not available")

except Exception as _e1:
    # ── Attempt 2: new mp.tasks (mediapipe >= 0.10) ───────
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as _mp_tasks_python
        from mediapipe.tasks.python import vision as _mp_vision
        from mediapipe.tasks.python.core import base_options as _base_options

        print(f"✅ MediaPipe {mp.__version__} loaded via Tasks API (HandLandmarker)")
        _USE_TASKS_API = True

        # ── Build a thin wrapper so apps.py code is unchanged ──

        class _HandLandmarkResult:
            """Mimics mp.solutions.hands result object."""
            def __init__(self, detection_result):
                self.multi_hand_landmarks = []
                if detection_result and detection_result.hand_landmarks:
                    for hand in detection_result.hand_landmarks:
                        lm_list = _LandmarkList(hand)
                        self.multi_hand_landmarks.append(lm_list)

        class _NormalizedLandmark:
            def __init__(self, lm):
                self.x = lm.x
                self.y = lm.y
                self.z = lm.z

        class _LandmarkList:
            def __init__(self, hand_landmarks):
                self.landmark = [_NormalizedLandmark(lm) for lm in hand_landmarks]

        class _DrawingUtils:
            @staticmethod
            def draw_landmarks(image, landmark_list, connections, style1=None, style2=None):
                h, w, _ = image.shape
                for lm in landmark_list.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)
                if connections:
                    for start_idx, end_idx in connections:
                        if start_idx < len(landmark_list.landmark) and end_idx < len(landmark_list.landmark):
                            start = landmark_list.landmark[start_idx]
                            end = landmark_list.landmark[end_idx]
                            pt1 = (int(start.x * w), int(start.y * h))
                            pt2 = (int(end.x * w), int(end.y * h))
                            cv2.line(image, pt1, pt2, (0, 200, 100), 2)

        class _DrawingStyles:
            @staticmethod
            def get_default_hand_landmarks_style():
                return None
            @staticmethod
            def get_default_hand_connections_style():
                return None

        # Connections list (same as mp.solutions.hands.HAND_CONNECTIONS)
        _HAND_CONNECTIONS = (
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        )

        class _HandsContextManager:
            """
            Wraps HandLandmarker to act as a context manager like mp.solutions.hands.Hands.
            Usage:  with mp_hands.Hands(model_complexity=0, ...) as hands:
                        image, results = mediapipe_detection(frame, hands)
            """
            def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
                options = _mp_vision.HandLandmarkerOptions(
                    base_options=_base_options.BaseOptions(model_asset_path=_HAND_LANDMARKER_MODEL),
                    running_mode=_mp_vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._detector = _mp_vision.HandLandmarker.create_from_options(options)

            def process(self, rgb_image):
                """Process an RGB numpy array and return results in legacy format."""
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                detection = self._detector.detect(mp_image)
                return _HandLandmarkResult(detection)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self._detector.close()

        class _HandsModule:
            """Mimics mp.solutions.hands module."""
            HAND_CONNECTIONS = _HAND_CONNECTIONS

            def Hands(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
                return _HandsContextManager(
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )

        mp_drawing = _DrawingUtils()
        mp_drawing_styles = _DrawingStyles()
        mp_hands = _HandsModule()

    except Exception as _e2:
        # ── Fallback: mock (no hand detection) ────────────────
        print(f"⚠️  MediaPipe not available, using mock (no hand detection). Error: {_e2}")
        USE_MOCK_MEDIAPIPE = True
        _USE_TASKS_API = False

        class MockHands:
            HAND_CONNECTIONS = ()
            def Hands(self, **kwargs):
                return self
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def process(self, image):
                class _R: multi_hand_landmarks = None
                return _R()

        class MockDrawingUtils:
            @staticmethod
            def draw_landmarks(image, landmarks, connections, style1=None, style2=None): pass

        class MockDrawingStyles:
            @staticmethod
            def get_default_hand_landmarks_style(): return None
            @staticmethod
            def get_default_hand_connections_style(): return None

        mp_drawing = MockDrawingUtils()
        mp_drawing_styles = MockDrawingStyles()
        mp_hands = MockHands()


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    if results and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


def extract_keypoints(results):
    if results and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten() \
                 if hand_landmarks else np.zeros(21 * 3)
            return rh
    return np.zeros(21 * 3)


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

actions = np.array(['HELLO','GOOD AFTERNOON','HOW ARE YOU','I AM FINE','HAD YOUR LUNCH','I ALSO HAD','OKAY','BYE',',','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','ALL THE BEST','THANK YOU'])

no_sequences = 30

sequence_length = 30
