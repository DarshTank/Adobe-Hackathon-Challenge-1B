# Use a specific Python version for reproducibility
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script into the container
COPY Process.py .

# Command to run your script when the container starts
CMD ["python", "Process.py"]