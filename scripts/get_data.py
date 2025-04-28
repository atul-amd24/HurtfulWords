import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path

# Set input and output paths
data_dir = Path("/Users/aravind/Premnisha/MS/dlh/sampled")
output_folder = Path("/Users/aravind/Premnisha/MS/dlh/HurtfulWords/data")
output_folder.mkdir(parents=True, exist_ok=True)

# Load CSVs
pats = pd.read_csv(data_dir / "PATIENTS.csv", parse_dates=["DOB", "DOD"])
adm = pd.read_csv(data_dir / "ADMISSIONS.csv", parse_dates=["ADMITTIME", "DEATHTIME", "DISCHTIME"])
notes = pd.read_csv(data_dir / "NOTEEVENTS.csv", parse_dates=["CHARTDATE", "CHARTTIME"])
diag = pd.read_csv(data_dir / "DIAGNOSES_ICD.csv")
icustays = pd.read_csv(data_dir / "ICUSTAYS.csv", parse_dates=["INTIME", "OUTTIME"])
icustays.columns = icustays.columns.str.lower()
icustays = icustays.set_index(['subject_id', 'hadm_id'])

print("✅ Patients:", len(pats))
print("✅ Admissions:", len(adm))
print("✅ Notes:", len(notes))
print("✅ Diagnoses:", len(diag))
print("✅ Columns on Patients table:", pats.columns)
print("✅ Columns on Admissions table:", adm.columns)
print("✅ Columns on notes table:", notes.columns)
print("✅ Columns on Diagnoses table:", diag.columns)


# Step 1: KFold assignment to patients
n_splits = 12
pats = pats.sample(frac=1, random_state=42).reset_index(drop=True)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for c, i in enumerate(kf.split(pats, groups=pats.GENDER)):
    pats.loc[i[1], 'fold'] = str(c)

# Step 2: Merge admissions with patients
df = pd.merge(pats, adm.rename(columns={"DIAGNOSIS": "adm_diag"}), on="SUBJECT_ID", how="inner")

# print("✅ After patients+admissions merge:", df.shape)
# print("✅ Columns before merging notes:", df.columns)

# Step 3: Merge death information
df['DOD_MERGED'] = df.apply(lambda row: row['DEATHTIME'] if pd.notnull(row['DEATHTIME']) else row['DOD'], axis=1)

print("✅ DOB non-null before notes merge:", df['DOB'].notna().sum())

# Step 4: Process notes
# print(f"Before merge with notes {notes['TEXT'].head(10)}")
# print(f"notes sum: {notes['TEXT'].isna().sum()}")
# print(f"notes empty: {(notes['TEXT'] == '').sum()}")

notes = notes[~notes['HADM_ID'].isnull()]
df_notes = notes

df['HADM_ID'] = df['HADM_ID'].astype(str).str.split('.').str[0]
df_notes['HADM_ID'] = df_notes['HADM_ID'].astype(str).str.split('.').str[0]
df_notes.rename(columns={'ROW_ID': 'NOTE_ID'}, inplace=True)

# print(df['HADM_ID'].unique()[:5])
# print(df_notes['HADM_ID'].unique()[:5])

df = pd.merge(df,df_notes, on='HADM_ID', how='left')

df_check = df[['CHARTDATE', 'DOB']].dropna()

# print("Rows with merged TEXT:", df['TEXT'].notna().sum())
# print("Total rows in df:", len(df))
# print(df['HADM_ID'].dtype)
# print(df_notes['HADM_ID'].dtype)

# df_test = pd.merge(df, df_notes[['HADM_ID', 'TEXT']], on='HADM_ID', how='inner')
# print(df_test['TEXT'].head())
print(f"After merge with notes {df_check.head(10)}")
print("✅ DOB non-null after notes merge:", df['DOB'].notna().sum())
print("✅ After notes merge:", df.shape)
# pd.set_option('display.max_colwidth', None)
# print("✅ notes table:", df_notes['TEXT'].head(1))
# print("✅ DF table:", df['TEXT'].head(1))

# print("✅ Columns after merging notes:", df.columns)

# Step 5: Fill missing ethnicity
df['ETHNICITY'].fillna('UNKNOWN/NOT SPECIFIED', inplace=True)

def clean_ethnicity(val):
    mappings = {
        'HISPANIC OR LATINO': 'HISPANIC/LATINO',
        'BLACK/AFRICAN AMERICAN': 'BLACK',
        'UNABLE TO OBTAIN': 'UNKNOWN/NOT SPECIFIED',
        'PATIENT DECLINED TO ANSWER': 'UNKNOWN/NOT SPECIFIED'
    }
    bases = ['WHITE', 'UNKNOWN/NOT SPECIFIED', 'BLACK', 'HISPANIC/LATINO', 'OTHER', 'ASIAN']
    if val in bases:
        return val
    elif val in mappings:
        return mappings[val]
    else:
        for b in bases:
            if b in str(val):
                return b
        return 'OTHER'

df['ETHNICITY_TO_USE'] = df['ETHNICITY'].apply(clean_ethnicity)

print("→ Missing CHARTDATE:", df['CHARTDATE'].isna().sum())
print("→ Missing DOB:", df['DOB'].isna().sum())
# Step: Fill missing CHARTDATE using DOB
df['CHARTDATE'] = df.apply(
    lambda row: row['CHARTDATE'] if pd.notnull(row['CHARTDATE']) else row.get('DOB', pd.NaT),
    axis=1
)

# print("✅ Missing CHARTDATE after fallback fill:", df['CHARTDATE'].isna().sum())

# df_check = df[['CHARTDATE', 'DOB']].dropna()
# print(df_check.head(10))
# Step 6: Convert date fields
df['CHARTDATE'] = pd.to_datetime(df['CHARTDATE'], errors='coerce')
df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
df['DOD'] = pd.to_datetime(df['DOD'], errors='coerce')

print("✅ BEFORE filtering CHARTDATE >= DOB:", df.shape)

# Step 7b: Filter out invalid or extreme dates before calculating age
df = df[(df['CHARTDATE'] >= df['DOB'])]
df = df[(df['CHARTDATE'].dt.year < 2200) & (df['DOB'].dt.year > 2000)]

# df = df[(df['CHARTDATE'] >= df['DOB']) & (df['CHARTDATE'] < pd.Timestamp('2160-01-01'))]
print("✅ After filtering CHARTDATE >= DOB:", df.shape)

# Step 8: Calculate age
df['AGE'] = (df['CHARTDATE'] - df['DOB']).dt.days / 365.24

# Step 9: Mark certain categories as fold='NA'
df.loc[df['CATEGORY'].isin(['Discharge summary', 'Echo', 'ECG']), 'fold'] = 'NA'

# Step 10: Merge diagnoses
diag = diag.groupby('HADM_ID').agg({'ICD9_CODE': lambda x: list(x)}).reset_index()
diag['HADM_ID'] = diag['HADM_ID'].astype(str)
df = pd.merge(df, diag, on='HADM_ID', how='left')

# Step 11: Map language field
df['LANGUAGE_TO_USE'] = df['LANGUAGE'].apply(lambda x: 'English' if x == 'ENGL' else ('Missing' if pd.isnull(x) else 'Other'))

# Step 12: Fill ICU stay ID
def fill_icustay(row):
    try:
        opts = icustays.xs((row['SUBJECT_ID'], row['HADM_ID']), drop_level=False)
    except KeyError:
        return None

    charttime = row['CHARTTIME'] if pd.notnull(row['CHARTTIME']) else row['CHARTDATE'] + pd.Timedelta(days=2)
    stay = opts[opts['intime'] <= charttime].sort_values(by='intime', ascending=True)
    return stay.iloc[-1]['icustay_id'] if not stay.empty else None

df['icustay_id'] = df[df['CATEGORY'].isin(['Discharge summary', 'Physician', 'Nursing', 'Nursing/other'])].apply(fill_icustay, axis=1)

# Step 13: Cap ages above 90
df.loc[df['AGE'] >= 90, 'AGE'] = 91.4
print("✅ Final dataframe shape before saving:", df.shape)

df.columns = df.columns.str.lower()  # lowercase columns before checking
df.rename(columns={
    'row_id_x': 'row_id',
    'subject_id_x': 'subject_id'
}, inplace=True)
print("✅ Columns:", df.columns)

# df.to_pickle(output_folder / "df_raw-full.pkl")
# print("✅ Processing complete. Saved to:", output_folder / "df_raw-full.pkl")

# Step 14: Save as csv
df.to_csv(output_folder / "df_raw-full.csv", index=False)
print("✅ Processing complete. Saved to:", output_folder / "df_raw-full.csv")