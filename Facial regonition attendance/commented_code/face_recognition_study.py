# Importing necessary libraries

# For interacting with the file system (creating directories, checking paths)
import os

# OpenCV library for image processing and face detection
import cv2

# NumPy for numerical operations and handling arrays/images
import numpy as np

# Pandas for working with CSV files and dataframes (used for attendance records)
import pandas as pd

# Joblib to save/load trained machine learning models
import joblib

# To get current date and time for logging attendance
from datetime import datetime, date

# K-Nearest Neighbors classifier for recognizing faces based on training data
from sklearn.neighbors import KNeighborsClassifier


# Define a class that encapsulates the entire face recognition-based attendance system
class FaceRecognitionSystem:
    def __init__(self):
        # Number of images to capture per user during training
        self.nimgs = 10

        # Load the pre-trained Haar Cascade model for face detection
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Path to save/load the trained face recognition model
        self.model_path = 'static/face_recognition_model.pkl'

        # Directory where user face images are stored for training
        self.face_dir = 'static/faces'

        # Directory to store attendance records
        self.attendance_dir = 'Attendance'

        # Generate today's date in format: month_day_year
        self.datetoday = date.today().strftime("%m_%d_%y")

        # Path to today's attendance CSV file
        self.csv_path = f'{self.attendance_dir}/Attendance-{self.datetoday}.csv'

        # Ensure all required directories exist
        self.ensure_directories()

        # Initialize the CSV file if it doesn't exist
        self.initialize_csv()


    # Method to create necessary directories if they don't exist
    def ensure_directories(self):
        for folder in [self.face_dir, 'static', self.attendance_dir]:
            if not os.path.isdir(folder):  # Check if directory exists
                os.makedirs(folder)       # Create it if it doesn't


    # Method to initialize the attendance CSV file
    def initialize_csv(self):
        if not os.path.exists(self.csv_path):  # If CSV doesn't exist
            with open(self.csv_path, 'w') as f:  # Open file in write mode
                f.write('Name,Roll,Time\n')      # Write header row


    # Method to detect faces in an image
    def extract_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5)  # Detect faces
        return faces  # Return list of detected face rectangles


    # Method to recognize a face from its feature array
    def identify_face(self, facearray):
        if os.path.exists(self.model_path):  # If model exists
            model = joblib.load(self.model_path)  # Load the trained model
            return model.predict(facearray)       # Predict the person's name
        return ["Unknown_0"]  # Default response if no model is found


    # Method to train the face recognition model using stored images
    def train_model(self):
        faces = []     # List to store flattened face images
        labels = []    # List to store corresponding labels (user names)

        # Loop through each user in the dataset
        for user in os.listdir(self.face_dir):
            # Loop through each image of that user
            for imgname in os.listdir(f'{self.face_dir}/{user}'):
                img = cv2.imread(f'{self.face_dir}/{user}/{imgname}')  # Read image
                resized = cv2.resize(img, (50, 50))  # Resize to uniform size
                faces.append(resized.ravel())        # Flatten and add to faces list
                labels.append(user)                  # Add label (username)

        # Only proceed if there are any faces collected
        if faces:
            model = KNeighborsClassifier(n_neighbors=5)  # Create KNN classifier
            model.fit(faces, labels)                     # Train the model
            joblib.dump(model, self.model_path)          # Save the trained model


    # Method to mark attendance for a recognized person
    def add_attendance(self, name):
        username, userid = name.split('_')  # Split name into user and roll number
        now = datetime.now().strftime("%H:%M:%S")  # Get current time
        df = pd.read_csv(self.csv_path)  # Load existing attendance data

        # If this roll number is not already in the file
        if int(userid) not in df['Roll'].values:
            # Create a new DataFrame row for this entry
            new_entry = pd.DataFrame([[username, int(userid), now]], columns=['Name', 'Roll', 'Time'])
            # Append new entry to the dataframe
            df = pd.concat([df, new_entry], ignore_index=True)
            # Save updated dataframe back to CSV
            df.to_csv(self.csv_path, index=False)


    # Method to retrieve attendance records
    def get_attendance(self):
        try:
            df = pd.read_csv(self.csv_path)  # Try to read CSV
            return df  # Return the dataframe
        except:
            # If file doesn't exist, return an empty dataframe with correct headers
            return pd.DataFrame(columns=['Name', 'Roll', 'Time'])