import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import time

# Initialize Mediapipe hands and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Variables for calculating FPS
pTime = 0
cTime = 0

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get coordinates of thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convert normalized coordinates to pixel coordinates
                h, w, _ = image.shape
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                # Draw circles on thumb and index finger tips
                cv2.circle(image, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)

                # Draw line between thumb and index finger tips
                cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 3)

                # Calculate distance between thumb and index finger tips
                length = math.hypot(index_x - thumb_x, index_y - thumb_y)

                # Convert length to volume level
                # Adjust these values according to your preference
                vol = np.interp(length, [20, 150], [min_vol, max_vol])
                vol_bar = np.interp(length, [20, 150], [400, 150])
                vol_perc = np.interp(length, [20, 150], [0, 100])

                # Set volume
                volume.SetMasterVolumeLevel(vol, None)

                # Draw volume bar
                cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
                cv2.rectangle(image, (50, int(vol_bar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(image, f'{int(vol_perc)} %', (40, 430), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
        pTime = cTime

        cv2.putText(image, f'FPS: {int(fps)}', (500, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        # Display the resulting image
        cv2.imshow('Gesture Volume Control', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
