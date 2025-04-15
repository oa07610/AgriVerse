# Use an official Python runtime as a base image
FROM python:3.11-slim


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .
# copy the wait script
COPY wait-for-db.sh /wait-for-db.sh
RUN chmod +x /wait-for-db.sh


# Expose the port that your Flask app runs on (commonly 5000)
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the command to start your app
#CMD ["flask", "run", "--host=0.0.0.0"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

