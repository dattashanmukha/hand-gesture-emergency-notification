# Import necessary libraries
import cv2
import numpy as np
import mediapipe as mp
import pywhatkit
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
WHATSAPP_NUMBER = '+919959363113'  # Replace with the recipient's WhatsApp number

# Function to send WhatsApp message
def send_whatsapp_message():
    try:
        pywhatkit.sendwhatmsg_instantly(phone_no=WHATSAPP_NUMBER, message="Emergency Detected! Help!", wait_time=50)
        print("WhatsApp message sent successfully.")
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Hardcoded gesture class names
class_names = ['fist', 'open', 'peace', 'thumbs_up', 'help']

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

                # Predict gesture
                try:
                    landmarks_array = np.array(landmarks).flatten()
                    if len(landmarks_array) == 42:  # Ensure expected input size
                        prediction = [0.1] * len(class_names)  # Mock prediction for demonstration
                        prediction[-1] = 0.8  # Mock confidence for 'help'
                        class_id = np.argmax(prediction)
                        class_name = class_names[class_id]
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    continue

                # Trigger actions if "help" gesture is detected
                if class_name.lower() == 'help':
                    print("Help gesture detected. Initiating emergency actions...")
                    send_whatsapp_message()

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
