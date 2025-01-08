# Import necessary libraries
import cv2
import numpy as np
import mediapipe as mp
import pywhatkit
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define recipient numbers
THUMB_UP_NUMBER = '+91 85208 94939'  # Replace with the recipient's WhatsApp number for thumb up
INDEX_UP_NUMBER = '+91 91103 81484'  # Replace with the recipient's WhatsApp number for index up

# Function to send WhatsApp message
def send_whatsapp_message(phone_no, message):
    try:
        pywhatkit.sendwhatmsg_instantly(phone_no=phone_no, message=message, wait_time=50)
        print(f"WhatsApp message sent successfully to {phone_no}.")
    except Exception as e:
        print(f"Error sending WhatsApp message to {phone_no}: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Helper function to check raised fingers
def detect_fingers(landmarks):
    fingers = []
    # Thumb: Compare tip (4) to MCP (2)
    fingers.append(landmarks[4][0] > landmarks[3][0])  # Assuming right hand; invert for left
    # Index: Compare tip (8) to PIP (6)
    fingers.append(landmarks[8][1] < landmarks[6][1])
    return fingers

# Main function
def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        class_name = ""

        # Process detected hand landmarks
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                # Extract landmarks
                landmarks = [[int(lm.x * w), int(lm.y * h)] for lm in hand_lms.landmark]

                # Draw landmarks and connections on the frame
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                # Detect raised fingers
                try:
                    fingers = detect_fingers(landmarks)
                    if fingers[0] and not fingers[1]:
                        class_name = 'thumbs_up'
                        print("Thumbs up gesture detected. Sending emergency message...")
                        send_whatsapp_message(THUMB_UP_NUMBER, "Emergency detected. Please help!!")
                    elif fingers[1] and not fingers[0]:
                        class_name = 'index_up'
                        print("Index finger up gesture detected. Sending emergency message...")
                        send_whatsapp_message(INDEX_UP_NUMBER, "Emergency detected. Please help!!")
                except Exception as e:
                    print(f"Error during finger detection: {e}")
                    continue

        # Display the prediction on the frame
        cv2.putText(frame, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Gesture Recognition", frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Entry point
if __name__ == "__main__":
    main()
