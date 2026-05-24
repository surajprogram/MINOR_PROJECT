import pandas as pd
from datetime import datetime
import os

# Get the directory of this script and construct the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(SCRIPT_DIR, "../data/attendance.xlsx")
COLUMNS = ["Name", "Date", "Time", "Faculty", "Department"]


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

    return {"status": "marked", "name": name}
