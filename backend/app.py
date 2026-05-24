from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
import os
import io
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Simple user storage (JSON file)
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")
FACULTY_FILE = os.path.join(os.path.dirname(__file__), "faculty_list.json")
ATTENDANCE_FILE = os.path.join(os.path.dirname(__file__), "data/attendance.xlsx")

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def load_faculty_list():
    if os.path.exists(FACULTY_FILE):
        with open(FACULTY_FILE, "r") as f:
            return json.load(f)
    return {}

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Attendance System API",
        "endpoints": {
            "/": "This endpoint",
            "/health": "Health check",
            "/start-attendance": "Start face recognition and mark attendance"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/api/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        
        if not data or not all(k in data for k in ["name", "email", "password"]):
            return jsonify({"message": "Missing required fields"}), 400
        
        users = load_users()
        
        if data["email"] in users:
            return jsonify({"message": "Email already registered"}), 400
        
        users[data["email"]] = {
            "name": data["name"],
            "email": data["email"],
            "password": data["password"]
        }
        save_users(users)
        
        return jsonify({"message": "Registration successful"}), 201
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        
        if not data or not all(k in data for k in ["email", "password"]):
            return jsonify({"message": "Missing email or password"}), 400
        
        users = load_users()
        user = users.get(data["email"])
        
        if not user or user["password"] != data["password"]:
            return jsonify({"message": "Invalid email or password"}), 401
        
        return jsonify({
            "message": "Login successful",
            "token": data["email"],
            "user": {"name": user["name"], "email": user["email"]}
        }), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route("/api/get-faculties", methods=["GET"])
def get_faculties():
    """Get list of faculties by department"""
    try:
        faculty_list = load_faculty_list()
        return jsonify(faculty_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/download-attendance", methods=["GET"])
def download_attendance():
    """Download attendance sheet as Excel file"""
    try:
        if not os.path.exists(ATTENDANCE_FILE):
            return jsonify({"message": "No attendance records found"}), 404

        from models.attendance import load_attendance_df

        df = load_attendance_df()
        if df.empty:
            return jsonify({"message": "No attendance records found"}), 404

        now = datetime.now()
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            groups = df.groupby(["Faculty", "Department"], dropna=False)
            used_sheets = set()

            for (faculty, department), group in groups:
                faculty_label = str(faculty).strip() or "Unknown Faculty"
                department_label = str(department).strip() or "Unknown Department"
                sheet_name = f"{faculty_label[:20]}_{department_label[:8]}".replace("/", "_").replace("\\", "_")
                sheet_name = sheet_name[:31]
                if sheet_name in used_sheets:
                    suffix = 1
                    base_name = sheet_name[:27]
                    while f"{base_name}_{suffix}" in used_sheets:
                        suffix += 1
                    sheet_name = f"{base_name}_{suffix}"
                used_sheets.add(sheet_name)

                group = group.sort_values(["Date", "Name"])
                group.to_excel(writer, sheet_name=sheet_name, index=False, startrow=4)

                worksheet = writer.sheets[sheet_name]
                worksheet.cell(row=1, column=1, value="Attendance Report")
                worksheet.cell(row=2, column=1, value=f"Teacher: {faculty_label}")
                worksheet.cell(row=3, column=1, value=f"Department: {department_label}")
                worksheet.cell(row=1, column=4, value=f"Exported On: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                worksheet.cell(row=2, column=4, value=f"Attendance Date: {now.strftime('%Y-%m-%d')}")

        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'attendance_{now.strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/test-camera", methods=["GET"])
def test_camera():
    try:
        from models.face_recognizer import check_camera
        ok, message = check_camera()
        if ok:
            return jsonify({"status": "ok", "message": message}), 200
        return jsonify({"status": "error", "message": message}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/start-attendance", methods=["GET"])
def start_attendance():
    try:
        # Import here to allow the app to start even if face_recognition_models is not available
        from models.face_recognizer import recognize_faces
        from models.attendance import mark_attendance
        
        faculty_name = request.args.get("faculty", "").strip()
        department = request.args.get("department", "").strip()
        # Require faculty and department to avoid creating anonymous/combined records
        if not faculty_name or not department:
            return jsonify({"message": "Missing faculty or department in request.", "success": False}), 400
        name = recognize_faces(timeout=10)  # 10 seconds timeout

        if name:
            result = mark_attendance(name, faculty=faculty_name, department=department)
            if result.get("status") == "already":
                return jsonify({
                    "message": f"Attendance already marked for {name} today.",
                    "success": True,
                    "alreadyMarked": True
                }), 200

            return jsonify({"message": f"Attendance marked for {name}", "success": True, "alreadyMarked": False}), 200
        else:
            return jsonify({"message": "No face detected. Please try again.", "success": False}), 400
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)