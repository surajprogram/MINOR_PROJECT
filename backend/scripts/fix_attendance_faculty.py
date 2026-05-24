import pandas as pd
import os

p = os.path.join(os.path.dirname(__file__), "..", "data", "attendance.xlsx")
if not os.path.exists(p):
    print('No attendance file at', p)
    raise SystemExit(1)

df = pd.read_excel(p, engine="openpyxl")
print('Before fill - unique faculty:', df['Faculty'].fillna('').unique())
print('Before fill - unique dept:', df['Department'].fillna('').unique())

# Fill missing values with explicit Unknown markers
df['Faculty'] = df['Faculty'].fillna('').astype(str).str.strip()
df['Department'] = df['Department'].fillna('').astype(str).str.strip()

missing_faculty_mask = df['Faculty'] == ''
missing_dept_mask = df['Department'] == ''

if missing_faculty_mask.any():
    df.loc[missing_faculty_mask, 'Faculty'] = 'Unknown Faculty'
if missing_dept_mask.any():
    df.loc[missing_dept_mask, 'Department'] = 'Unknown Department'

# Save backup
bak = p + '.bak'
print('Creating backup at', bak)
if not os.path.exists(bak):
    df.to_excel(bak, index=False, engine='openpyxl')

# Save fixed file
df.to_excel(p, index=False, engine='openpyxl')
print('After fill - unique faculty:', df['Faculty'].unique())
print('After fill - unique dept:', df['Department'].unique())
print('Done.')
