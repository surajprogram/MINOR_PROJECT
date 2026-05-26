import pandas as pd
from datetime import datetime
import os

# Get the directory of this script and construct the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(SCRIPT_DIR, "../data/attendance.xlsx")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "../data/attendance_reports")
COLUMNS = ["Name", "Date", "Time", "Faculty", "Department"]


def ensure_reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def get_faculty_report_path(faculty, department, date):
    faculty_label = str(faculty).strip() or "UnknownFaculty"
    department_label = str(department).strip() or "UnknownDepartment"
    safe_faculty = "".join(c for c in faculty_label if c.isalnum() or c in ["_", "-"]).replace(" ", "_")
    safe_department = "".join(c for c in department_label if c.isalnum() or c in ["_", "-"]).replace(" ", "_")
    filename = f"attendance_{safe_faculty}_{safe_department}_{date}.xlsx"
    return os.path.join(REPORTS_DIR, filename)


def save_faculty_report(df, faculty, department, date):
    ensure_reports_dir()
    faculty = (faculty or "").strip()
    department = (department or "").strip()
    if not faculty or not department:
        return

    report_df = df[
        (df["Faculty"].astype(str).str.strip().str.lower() == faculty.lower()) &
        (df["Department"].astype(str).str.strip().str.lower() == department.lower()) &
        (df["Date"] == date)
    ]

    if report_df.empty:
        return

    report_path = get_faculty_report_path(faculty, department, date)
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        report_df.to_excel(writer, index=False, startrow=4)
        worksheet = writer.sheets["Sheet1"]
        worksheet.cell(row=1, column=1, value="Attendance Report")
        worksheet.cell(row=2, column=1, value=f"Faculty: {faculty}")
        worksheet.cell(row=3, column=1, value=f"Department: {department}")
        worksheet.cell(row=1, column=4, value=f"Date: {date}")


def load_attendance_df():
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
        except Exception:
            df = pd.DataFrame(columns=COLUMNS)
    else:
        df = pd.DataFrame(columns=COLUMNS)

    for column in COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[COLUMNS].fillna("")
    return df


def save_attendance_df(df):
    df.to_excel(file_path, index=False, engine="openpyxl")


def mark_attendance(name, faculty=None, department=None):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    faculty = (faculty or "").strip()
    department = (department or "").strip()

    df = load_attendance_df()
    duplicate = (
        df["Name"].astype(str).str.strip().str.lower() == str(name).strip().lower()
    ) & (
        df["Date"] == date
    ) & (
        df["Faculty"].astype(str).str.strip().str.lower() == faculty.lower()
    ) & (
        df["Department"].astype(str).str.strip().str.lower() == department.lower()
    )

    if duplicate.any():
        return {"status": "already", "name": name}

    new_entry = {
        "Name": name,
        "Date": date,
        "Time": time,
        "Faculty": faculty,
        "Department": department,
    }

    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    save_attendance_df(df)
    save_faculty_report(df, faculty, department, date)

    return {"status": "marked", "name": name}
