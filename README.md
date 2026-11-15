# 🎓 AI-Based College Attendance System Using Face Recognition

## 📌 About the Project

The **AI-Based College Attendance System** is a smart attendance management application that uses Artificial Intelligence and Face Recognition technology to make the attendance process easier and faster.

In traditional college attendance systems, teachers need to manually call students' names or maintain attendance registers. This process takes time and sometimes leads to mistakes.

To solve this problem, we developed a system that uses a camera to detect and recognize students' faces. When a registered student is successfully identified, the system can record their attendance automatically.

The main purpose of this project is to reduce manual work, save time, and make college attendance management smarter.

## ✨ Key Features

- **Face Detection:** Detects human faces using a camera.
- **Face Recognition:** Identifies registered students using their facial features.
- **Automatic Attendance:** Records attendance after successful face recognition.
- **Real-Time Processing:** Processes camera input for face detection and recognition.
- **Student Management:** Supports working with registered student information.
- **Attendance Records:** Helps maintain attendance information.
- **Easy-to-Use Interface:** Provides a frontend interface for interacting with the application.

## ⚙️ How Does It Work?

The system follows a simple process:

1. **Student Registration:** Student information and face images are added to the system.
2. **Camera Activation:** The camera captures the student's face.
3. **Face Detection:** The system detects the face from the captured image.
4. **Face Recognition:** The detected face is compared with registered student information.
5. **Student Identification:** If a matching face is found, the student is identified.
6. **Attendance Recording:** The system records attendance with the relevant date and time.

## 🛠️ Technologies Used

### Frontend
- React.js
- Vite
- JavaScript
- HTML
- CSS

### Backend
- Python
- Flask
- Flask-CORS

### Artificial Intelligence & Computer Vision
- Face Recognition
- OpenCV
- Dlib
- MediaPipe
- NumPy

### Data Handling
- Pandas
- OpenPyXL

## 📂 Project Structure

```text
MINOR_PROJECT/
│
├── backend/
│   └── Backend application files
│
├── frontend/
│   └── Frontend application files
│
├── requirement.txt
├── vite.config.js
└── README.md
```

## 🚀 Installation and Setup

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/surajprogram/MINOR_PROJECT.git
```

### Step 2: Open the Project

```bash
cd MINOR_PROJECT
```

### Step 3: Create a Python Virtual Environment

```bash
python -m venv myenv
```

### Step 4: Activate Virtual Environment

For Windows:

```bash
myenv\Scripts\activate
```

### Step 5: Install Python Dependencies

```bash
pip install -r requirement.txt
```

### Step 6: Start the Backend

Open the backend folder and run the Python application using its main Flask application file.

For example, if the entry file is `app.py`:

```bash
python app.py
```

### Step 7: Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed in your terminal to access the application.

## 🎯 Project Objectives

- Automate the traditional attendance process.
- Reduce the time required for taking attendance.
- Use AI to recognize registered students.
- Reduce manual attendance-related work.
- Make attendance records easier to manage.

## 🔮 Future Improvements

Some improvements that can be introduced in the future:

- Live attendance dashboard.
- Attandance Save in database 
- Attendance reports in Excel and PDF format.
- Student attendance percentage calculation.
- Email or SMS notifications.
- Advanced face verification and liveness detection.
- Cloud-based attendance management.
- Mobile application integration.

## 🔐 Privacy and Security

Student facial information is sensitive and should be handled responsibly. The system should use facial data only with appropriate consent and authorization. Access to student information and attendance records should be restricted to authorized users.

## 👨‍💻 Developer

**Developed by Suraj Gupta**

GitHub: [@surajprogram](https://github.com/surajprogram)

## ⭐ Support

If you find this project interesting, consider giving the repository a star on GitHub!

**Thank you for visiting my project! ❤️**
