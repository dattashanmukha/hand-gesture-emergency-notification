# Hand Gesture Recognition with WhatsApp Alerts

## Overview
This project uses computer vision and machine learning to recognize specific hand gestures and send emergency WhatsApp alerts to designated phone numbers. The system captures live video from the webcam, detects hand gestures using MediaPipe, and sends predefined emergency messages via WhatsApp using `pywhatkit`.

## Features
- Detects "Thumbs Up" and "Pointing with Index Finger" gestures.
- Sends a custom emergency message to a designated phone number when a specific gesture is detected.
- The system runs in real-time using webcam input.
- Uses `pywhatkit` to schedule WhatsApp messages for immediate sending.

## Prerequisites
- Python 3.11.3
- OpenCV
- NumPy
- MediaPipe
- TensorFlow
- A phone number with WhatsApp account


## Installation
1. **Clone the repository:**
```
git clone https://github.com/dattashanmukha/hand-gesture-emergency-notification.git
```

2. **Install the required packages:**
```
pip install -r requirement.txt
```

## Requirements
To run this project, you'll need the following libraries:

- `opencv-python` for webcam and image processing.
- `mediapipe` for hand tracking and gesture recognition.
- `pywhatkit` for sending WhatsApp messages.

You can install the required libraries using pip:

```bash
pip install opencv-python mediapipe pywhatkit
```

## Usage
1. **Run the `hand_gesture_detection.py` file:**
```
python hand_gesture_detection.py

```

2. **Ensure your webcam is connected and functioning properly.**

3. **Show the "help" gesture to trigger the emergency notification.**



## Configuration
- Update the `phone_number` variable with the phone number you want to send the WhatsApp message to


## Customization

- You can customize the gestures by training your own gesture recognition model and updating the `model` variable in `main.py` with your model.
- Adjust the confidence threshold for gesture detection by modifying the `min_detection_confidence` variable in `main.py`.
- You can also customize the code to recognize a specific number of hands.


If you encounter any issues or have suggestions for improvements, please [open an issue](https://github.com/your-username/hand-gesture-emergency-notification/issues) or [create a pull request](https://github.com/your-username/hand-gesture-emergency-notification/pulls).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---



