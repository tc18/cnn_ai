# Use official PyTorch image as base
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir flask torch torchvision pillow numpy opencv-python

# Expose the port the app runs on
EXPOSE 5001

# Run the application
CMD ["python", "app.py"]

# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]