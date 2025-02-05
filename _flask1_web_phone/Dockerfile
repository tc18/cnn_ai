# Use official PyTorch image as base
FROM python:3.8-slim-buster
# FROM pytorch/pytorch:latest

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

# Set the working directory
WORKDIR /app

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5001

# Run the Flask app
CMD ["python", "app.py"]

# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]