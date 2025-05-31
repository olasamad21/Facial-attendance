import os
import cv2
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, date
from sklearn.neighbors import KNeighborsClassifier

class FaceRecognitionSystem:
    def __init__(self):
        self.nimgs = 10
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model_path = 'static/face_recognition_model.pkl'
        self.face_dir = 'static/faces'
        self.attendance_dir = 'Attendance'
        self.datetoday = date.today().strftime("%m_%d_%y")
        self.csv_path = f'{self.attendance_dir}/Attendance-{self.datetoday}.csv'
        self.ensure_directories()
        self.initialize_csv()

    def ensure_directories(self):
        for folder in [self.face_dir, 'static', self.attendance_dir]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

    def initialize_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write('Name,Roll,Time\n')

    def extract_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5)
        return faces

    def identify_face(self, facearray):
        if os.path.exists(self.model_path):
            model = joblib.load(self.model_path)
            return model.predict(facearray)
        return ["Unknown_0"]

    def train_model(self):
        faces = []
        labels = []
        for user in os.listdir(self.face_dir):
            for imgname in os.listdir(f'{self.face_dir}/{user}'):
                img = cv2.imread(f'{self.face_dir}/{user}/{imgname}')
                resized = cv2.resize(img, (50, 50))
                faces.append(resized.ravel())
                labels.append(user)
        if faces:
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(faces, labels)
            joblib.dump(model, self.model_path)

    def add_attendance(self, name):
        username, userid = name.split('_')
        now = datetime.now().strftime("%H:%M:%S")
        df = pd.read_csv(self.csv_path)
        if int(userid) not in df['Roll'].values:
            new_entry = pd.DataFrame([[username, int(userid), now]], columns=['Name', 'Roll', 'Time'])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(self.csv_path, index=False)

    def get_attendance(self):
        try:
            df = pd.read_csv(self.csv_path)
            return df
        except:
            return pd.DataFrame(columns=['Name', 'Roll', 'Time'])