# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install them
COPY FinSight-app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (including model.pkl and features.pkl)
COPY FinSight-app/ .

# Expose the port the app runs on
EXPOSE 10000

# Command to run the application
CMD gunicorn main:app --bind 0.0.0.0:$PORT