# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN apt-get update && apt-get install -y bash
RUN pip install --no-cache-dir -r requirements.txt


# Copy the entire application folder to the container
COPY . .

# Copy the model files (ensure the path is correct)
COPY artifacts/best_model/best_model /artifacts/best_model/best_model

# Expose the port on which the Flask app will run
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
