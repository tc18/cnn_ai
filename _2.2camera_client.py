import cv2
import numpy as np
import requests

API_URL = "http://127.0.0.1:5001/predict"

def preprocess_image(image):
    """ Convert OpenCV image to format suitable for the model """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    _, thresh = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY_INV)
    return thresh

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a box where user should write the digit
    height, width = frame.shape[:2]
    # x, y, w, h = width//3, height//3, width//3, height//3
    x, y, w, h = 900//3, 900//3, 900//3, 900//3
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract the region of interest (ROI)
    roi = frame[y:y+h, x:x+w]
    processed = preprocess_image(roi)

    # Show processed image in another window
    cv2.imshow("Processed Image", processed)

    # Convert to bytes and send to API
    _, img_encoded = cv2.imencode(".png", processed)
    response = requests.post(API_URL, files={"file": img_encoded.tobytes()})

    if response.status_code == 200:
        result = response.json()
        prediction = result.get("prediction", "Error")
        cv2.putText(frame, f"Predicted: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Live Digit Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
