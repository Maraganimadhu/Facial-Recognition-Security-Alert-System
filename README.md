# Facial-Recognition-Security-Alert-System

This project is a Python-based security system that uses facial recognition to detect and identify individuals. When an unknown person is detected, the system sends an alert message and stores the record in the database. It is ideal for enhancing home or workplace security using OpenCV and Python.

🔍 Features
Real-time facial recognition using OpenCV and LBPH algorithm

Capture and register new faces via GUI

Alert system on unknown face detection

Store records of visits with timestamp

User interface using HTML and Flask

Integrated login and registration pages

Responsive design with background image and navigation

🛠️ Tech Stack
Frontend: HTML, CSS, JavaScript

Backend: Python, Flask

Database: SQLite

Libraries: OpenCV, NumPy

#Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the application:

bash
Copy
Edit
python main.py
Open your browser and visit http://localhost:5000

📷 How It Works
The camera captures frames in real-time.

If a known face is detected, it grants access.

If an unknown face is found, an alert (e.g. via SMS/email) can be triggered.

You can register new users from the UI.

📌 Notes
Make sure your webcam is connected and working.

You need to create a dataset and train the recognizer before using recognition.

Background images must be correctly referenced in static/ for GitHub Pages to display them properly.

📄 License
This project is open source and free to use under the MIT License.

