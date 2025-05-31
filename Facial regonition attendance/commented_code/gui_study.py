# Import operating system interface for file/directory operations
import os
# Import OpenCV library for computer vision operations (camera access, image processing)
import cv2
# Import datetime for timestamp handling in attendance records
from datetime import datetime
# Import Tkinter components for GUI creation
from tkinter import Tk, Label, Button, Entry, StringVar, messagebox, Frame, LabelFrame
# Import custom face recognition system (separate module - not shown in this code)
from face_recognition_system import FaceRecognitionSystem


# Define main application class for the GUI interface
class FaceApp:
    # Constructor method - initializes the application when object is created
    def __init__(self, root):
        # Store reference to main window
        self.root = root
        # Create instance of face recognition system (handles ML operations)
        self.system = FaceRecognitionSystem()
        # Create StringVar for username input (Tkinter variable for two-way binding)
        self.username = StringVar()
        # Create StringVar for user ID input (Tkinter variable for two-way binding)
        self.userid = StringVar()

        # Set window title that appears in title bar
        root.title("Face Recognition Attendance System")
        # Set window size to 400x300 pixels
        root.geometry("400x300")
        # Disable window resizing (fixed size)
        root.resizable(False, False)

        # Create main container frame with padding around edges
        main_frame = Frame(root, padx=20, pady=20)
        # Pack frame to fill entire window space
        main_frame.pack(expand=True, fill='both')

        # Create labeled frame for user input section with internal padding
        input_frame = LabelFrame(main_frame, text="User Details", padx=10, pady=10)
        # Pack input frame to fill horizontal space with bottom margin
        input_frame.pack(fill='x', pady=(0, 10))

        # Create "Username:" label in grid position (0,0), right-aligned
        Label(input_frame, text="Username:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        # Create username entry field in grid position (0,1), 25 characters wide
        Entry(input_frame, textvariable=self.username, width=25).grid(row=0, column=1, padx=5, pady=5)

        # Create "User ID:" label in grid position (1,0), right-aligned
        Label(input_frame, text="User ID:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        # Create user ID entry field in grid position (1,1), 25 characters wide
        Entry(input_frame, textvariable=self.userid, width=25).grid(row=1, column=1, padx=5, pady=5)

        # Create labeled frame for action buttons with internal padding
        action_frame = LabelFrame(main_frame, text="Actions", padx=10, pady=10)
        # Pack action frame to fill horizontal space
        action_frame.pack(fill='x')

        # Set consistent button width for uniform appearance
        btn_width = 15
        # Create "Add User" button that calls add_user method when clicked
        Button(action_frame, text="Add User", command=self.add_user, width=btn_width).pack(pady=5)
        # Create "Start Attendance" button that calls start_attendance method when clicked
        Button(action_frame, text="Start Attendance", command=self.start_attendance, width=btn_width).pack(pady=5)
        # Create "Show Attendance" button that calls show_attendance method when clicked
        Button(action_frame, text="Show Attendance", command=self.show_attendance, width=btn_width).pack(pady=5)

    # Method to add new user to the system
    def add_user(self):
        # Get username from input field and remove whitespace
        name = self.username.get().strip()
        # Try to convert user ID to integer, handle invalid input
        try:
            uid = int(self.userid.get().strip())
        # If conversion fails, show error and exit method
        except:
            messagebox.showerror("Invalid ID", "User ID must be a number.")
            return

        # Create folder path using face directory, username, and user ID
        folder = f'{self.system.face_dir}/{name}_{uid}'
        # Check if user folder already exists
        if os.path.exists(folder):
            # Show info message if user already exists
            messagebox.showinfo("Exists", "User already exists.")
            return
        # Create new directory for user's face images
        os.makedirs(folder)

        # Initialize camera capture (device 0 = default camera)
        cap = cv2.VideoCapture(0)
        # Initialize counters: i = saved images, j = frame counter
        i, j = 0, 0
        # Start infinite loop for image capture
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            # Skip if frame reading failed
            if not ret:
                continue
            # Extract face locations from current frame
            faces = self.system.extract_faces(frame)
            # Process each detected face
            for (x, y, w, h) in faces:
                # Save image every 5th frame (j % 5 == 0) to reduce similar images
                if j % 5 == 0:
                    # Save cropped face image to user folder
                    cv2.imwrite(f'{folder}/{name}_{i}.jpg', frame[y:y + h, x:x + w])
                    # Increment saved image counter
                    i += 1
                # Increment frame counter
                j += 1
                # Draw blue rectangle around detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Display count of captured images on frame
                cv2.putText(frame, f'Images Captured: {i}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Show camera feed with face detection overlay
            cv2.imshow("Adding User", frame)
            # Exit loop when enough images captured or ESC key pressed (ASCII 27)
            if i >= self.system.nimgs or cv2.waitKey(1) == 27:
                break
        # Release camera resource
        cap.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()

        # Train the machine learning model with new user data
        self.system.train_model()
        # Show success message to user
        messagebox.showinfo("Success", f"User {name} added successfully.")

    # Method to start attendance recognition process
    def start_attendance(self):
        # Check if trained model exists
        if not os.path.exists(self.system.model_path):
            # Show warning if no model found
            messagebox.showwarning("No Model", "No trained model found. Add users first.")
            return

        # Initialize camera capture
        cap = cv2.VideoCapture(0)
        # Record start time for timeout functionality
        start_time = datetime.now()
        # Set timeout duration to 10 seconds
        timeout = 10

        # Continue until timeout reached
        while (datetime.now() - start_time).seconds < timeout:
            # Read frame from camera
            ret, frame = cap.read()
            # Skip if frame reading failed
            if not ret:
                continue
            # Extract face locations from current frame
            faces = self.system.extract_faces(frame)
            # Process each detected face
            for (x, y, w, h) in faces:
                # Resize face to 50x50 and flatten for model input
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50)).reshape(1, -1)
                # Identify person using trained model (returns name)
                name = self.system.identify_face(face)[0]
                # Record attendance for identified person
                self.system.add_attendance(name)
                # Display person's name above their face
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Draw green rectangle around recognized face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Show attendance recognition window
            cv2.imshow("Attendance", frame)
            # Exit if ESC key pressed
            if cv2.waitKey(1) == 27:
                break
        # Release camera resource
        cap.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        # Show completion message
        messagebox.showinfo("Done", "Attendance Process Completed")

    # Method to display attendance records
    def show_attendance(self):
        # Get attendance data as pandas DataFrame
        df = self.system.get_attendance()
        # Check if any attendance records exist
        if df.empty:
            # Show message if no records found
            messagebox.showinfo("Attendance", "No attendance records yet.")
            return

        # Calculate optimal column widths for display formatting
        cols = df.columns.tolist()
        # Width = max of (column name length, max data length) + 4 for padding
        widths = {col: max(len(str(col)), df[col].astype(str).str.len().max()) + 4 for col in cols}

        # Build formatted string representation of data
        lines = []
        # Create header row with right-aligned column names
        lines.append("".join(str(col).rjust(widths[col]) for col in cols))
        # Create separator line using dashes
        lines.append("".join("-" * widths[col] for col in cols))
        # Add each data row with proper alignment
        for _, row in df.iterrows():
            lines.append("".join(str(row[col]).rjust(widths[col]) for col in cols))
            # Add empty line between records for readability
            lines.append("")

        # Import additional Tkinter components for display window
        from tkinter import Text, Scrollbar, VERTICAL, END, Toplevel
        # Create new popup window
        win = Toplevel(self.root)
        # Set popup window title
        win.title("Attendance Records")
        # Set popup window size
        win.geometry("600x400")

        # Create frame container for text widget
        frame = Frame(win)
        # Pack frame to fill entire popup window
        frame.pack(expand=True, fill='both', padx=10, pady=10)

        # Create text widget with monospace font (preserves alignment)
        text = Text(frame, font=('Courier', 10), wrap='none')
        # Create vertical scrollbar for text widget
        scroll = Scrollbar(frame, orient=VERTICAL, command=text.yview)
        # Connect scrollbar to text widget
        text.configure(yscrollcommand=scroll.set)

        # Pack text widget to fill space
        text.pack(side='left', fill='both', expand=True)
        # Pack scrollbar on right side
        scroll.pack(side='right', fill='y')

        # Insert formatted attendance data into text widget
        text.insert(END, "\n".join(lines))
        # Make text widget read-only
        text.config(state='disabled')

        # Create close button for popup window
        Button(win, text="Close", command=win.destroy).pack(pady=5)
        # Make popup window modal (blocks interaction with main window)
        win.grab_set()