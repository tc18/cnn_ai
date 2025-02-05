import cv2
import numpy as np
import torch

# Load the trained model
model = CNN()
model.load_state_dict(torch.load("cnn_mnist.pth"))
model.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    tensor_image = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    output = model(tensor_image)
    _, predicted = torch.max(output, 1)
    
    cv2.putText(frame, f"Predicted: {predicted.item()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Handwritten Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
