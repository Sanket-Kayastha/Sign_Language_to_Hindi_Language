import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for real-time video processing
    max_num_hands=2,  # Detect up to two hands
    min_detection_confidence=0.7,  # Increase confidence threshold for detection
    min_tracking_confidence=0.7)

# Define label dictionary with both English and Hindi characters
labels_dict = {
    0: ('A', 'अ'), 1: ('B', 'ब'), 2: ('C', 'क'), 3: ('D', 'ड'), 4: ('E', 'इ'), 5: ('F', 'फ'),
    6: ('G', 'ग'), 7: ('H', 'ह'), 8: ('I', 'ई'), 9: ('J', 'ज'), 10: ('K', 'क'), 11: ('L', 'ल'),
    12: ('M', 'म'), 13: ('N', 'न'), 14: ('O', 'ओ'), 15: ('P', 'प'), 16: ('Q', 'क्यु'), 17: ('R', 'र'),
    18: ('S', 'स'), 19: ('T', 'त'), 20: ('U', 'उ'), 21: ('V', 'व'), 22: ('W', 'डब्ल्यू'), 23: ('X', 'एक्स'),
    24: ('Y', 'य')
}

formed_word_english = ""
formed_word_hindi = ""
last_predicted_time = time.time()
prediction_buffer = deque(maxlen=10)

# Define a function to draw text with Pillow
def draw_text_with_pillow(image, text, position, font_path, font_size=30, color=(255, 0, 0)):
    # Convert OpenCV image (BGR) to PIL image (RGB)
    image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_pil)

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Use a truetype font
    font = ImageFont.truetype(font_path, font_size)

    # Draw text
    draw.text(position, text, font=font, fill=color)

    # Convert PIL image (RGB) back to OpenCV image (BGR)
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image_bgr

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        # Ensure data_aux has the correct number of features (84 in this case)
        while len(data_aux) < 84:
            data_aux.extend([0, 0])  # Padding with zeros if fewer landmarks detected
        if len(data_aux) > 84:
            data_aux = data_aux[:84]  # Truncate if too many landmarks detected

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        predicted_character = ""
        if time.time() - last_predicted_time > 1:
            prediction = model.predict([np.asarray(data_aux)])
            prediction_buffer.append(prediction[0])
            
            # Use a prediction buffer to stabilize predictions
            if len(prediction_buffer) == prediction_buffer.maxlen:
                most_common_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                predicted_character = labels_dict[int(most_common_prediction)]
                formed_word_english += predicted_character[0]
                formed_word_hindi += predicted_character[1]
                prediction_buffer.clear()
                last_predicted_time = time.time()
                print(predicted_character)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        if predicted_character:
            cv2.putText(frame, f"{predicted_character[0]} ({predicted_character[1]})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the formed word on the frame using Pillow
    text_to_display = f"Word: {formed_word_english} ({formed_word_hindi})"
    frame = draw_text_with_pillow(frame, text_to_display, (10, H - 50), font_path="./NotoSansDevanagari-Regular.ttf", font_size=30, color=(255, 0, 0))

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('\r'):  # Press Enter key to clear the formed word
        formed_word_english = ""
        formed_word_hindi = ""

cap.release()
cv2.destroyAllWindows()
