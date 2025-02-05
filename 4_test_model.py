# Get a sample test image
test_image, test_label = trainset[1]
with torch.no_grad():
    output = model(test_image.unsqueeze(0))
    predicted = torch.argmax(output, dim=1)

print(f"Predicted Label: {predicted.item()}, Actual Label: {test_label}")
