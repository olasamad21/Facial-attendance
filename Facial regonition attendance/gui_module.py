import os
import cv2
from datetime import datetime
from tkinter import Tk, Label, Button, Entry, StringVar, messagebox, Frame, LabelFrame
# Import custom face recognition system (separate module - handles ML operations)
from face_recognition_system import FaceRecognitionSystem


class FaceApp:
    def __init__(self, root):
        self.root = root
        # Create instance of face recognition system (handles ML operations)
        self.system = FaceRecognitionSystem()
        # Create StringVar for two-way binding with GUI inputs
        self.username = StringVar()
        self.userid = StringVar()

        root.title("Face Recognition Attendance System")
        root.geometry("400x300")
        root.resizable(False, False)

        main_frame = Frame(root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')

        input_frame = LabelFrame(main_frame, text="User Details", padx=10, pady=10)
        input_frame.pack(fill='x', pady=(0, 10))

        Label(input_frame, text="Username:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        Entry(input_frame, textvariable=self.username, width=25).grid(row=0, column=1, padx=5, pady=5)

        Label(input_frame, text="User ID:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        Entry(input_frame, textvariable=self.userid, width=25).grid(row=1, column=1, padx=5, pady=5)

        action_frame = LabelFrame(main_frame, text="Actions", padx=10, pady=10)
        action_frame.pack(fill='x')

        btn_width = 15
        Button(action_frame, text="Add User", command=self.add_user, width=btn_width).pack(pady=5)
        Button(action_frame, text="Start Attendance", command=self.start_attendance, width=btn_width).pack(pady=5)
        Button(action_frame, text="Show Attendance", command=self.show_attendance, width=btn_width).pack(pady=5)

    def add_user(self):
        name = self.username.get().strip()
        # Validate user ID input - must be numeric
        try:
            uid = int(self.userid.get().strip())
        except:
            messagebox.showerror("Invalid ID", "User ID must be a number.")
            return

        # Create folder path using face directory, username, and user ID
        folder = f'{self.system.face_dir}/{name}_{uid}'
        if os.path.exists(folder):
            messagebox.showinfo("Exists", "User already exists.")
            return
        os.makedirs(folder)

        # Initialize camera capture (device 0 = default camera)
        cap = cv2.VideoCapture(0)
        # Initialize counters: i = saved images, j = frame counter
        i, j = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            faces = self.system.extract_faces(frame)
            for (x, y, w, h) in faces:
                # Save image every 5th frame to reduce similar images and improve training diversity
                if j % 5 == 0:
                    cv2.imwrite(f'{folder}/{name}_{i}.jpg', frame[y:y + h, x:x + w])
                    i += 1
                j += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'Images Captured: {i}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Adding User", frame)
            # Exit when enough images captured or ESC key pressed
            if i >= self.system.nimgs or cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

        # Train the ML model with newly added user data
        self.system.train_model()
        messagebox.showinfo("Success", f"User {name} added successfully.")

    def start_attendance(self):
        # Check if trained model exists before starting recognition
        if not os.path.exists(self.system.model_path):
            messagebox.showwarning("No Model", "No trained model found. Add users first.")
            return

        cap = cv2.VideoCapture(0)
        start_time = datetime.now()
        # Set timeout duration to prevent infinite camera access
        timeout = 10

        while (datetime.now() - start_time).seconds < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            faces = self.system.extract_faces(frame)
            for (x, y, w, h) in faces:
                # Resize face to 50x50 and flatten for model input (standard preprocessing)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50)).reshape(1, -1)
                # Identify person using trained model
                name = self.system.identify_face(face)[0]
                self.system.add_attendance(name)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Done", "Attendance Process Completed")

    def show_attendance(self):
        df = self.system.get_attendance()
        if df.empty:
            messagebox.showinfo("Attendance", "No attendance records yet.")
            return

        # Calculate optimal column widths for proper data alignment
        cols = df.columns.tolist()
        widths = {col: max(len(str(col)), df[col].astype(str).str.len().max()) + 4 for col in cols}

        # Build formatted string representation with proper alignment
        lines = []
        lines.append("".join(str(col).rjust(widths[col]) for col in cols))
        lines.append("".join("-" * widths[col] for col in cols))
        for _, row in df.iterrows():
            lines.append("".join(str(row[col]).rjust(widths[col]) for col in cols))
            lines.append("")

        from tkinter import Text, Scrollbar, VERTICAL, END, Toplevel
        win = Toplevel(self.root)
        win.title("Attendance Records")
        win.geometry("600x400")

        frame = Frame(win)
        frame.pack(expand=True, fill='both', padx=10, pady=10)

        # Use monospace font to preserve column alignment
        text = Text(frame, font=('Courier', 10), wrap='none')
        scroll = Scrollbar(frame, orient=VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scroll.set)

        text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        text.insert(END, "\n".join(lines))
        # Make text widget read-only to prevent user modification
        text.config(state='disabled')

        Button(win, text="Close", command=win.destroy).pack(pady=5)
        # Make popup window modal to block interaction with main window
        win.grab_set()