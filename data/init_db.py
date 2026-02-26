import sqlite3
import pandas as pd
import json

DB_PATH = "medicines.db"
CSV_PATH = "Medicine_Details.csv"

# ── Read CSV and clean data ───────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Remove extra spaces from column names
df.columns = [col.strip() for col in df.columns]

# Convert review percentages to integers
for col in ["Excellent Review %", "Average Review", "Poor Review %"]:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "")
            .str.strip()
            .astype(int)
        )

# Convert Side_effects from comma-separated string to list
if "Side_effects" in df.columns:
    df["Side_effects"] = df["Side_effects"].astype(str).apply(
        lambda x: [s.strip() for s in x.split(",") if s.strip()]
    )

# ── Create SQLite database ────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS medicines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    composition TEXT,
    uses TEXT,
    side_effects TEXT,          -- stored as JSON list
    image_url TEXT,
    manufacturer TEXT,
    excellent INTEGER DEFAULT 0,
    average   INTEGER DEFAULT 0,
    poor      INTEGER DEFAULT 0
)
''')

for _, row in df.iterrows():
    side_json = json.dumps(row["Side_effects"])
    c.execute('''
        INSERT OR REPLACE INTO medicines 
        (name, composition, uses, side_effects, image_url, manufacturer, excellent, average, poor)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        row["Medicine Name"],
        row.get("Composition", ""),
        row.get("Uses", ""),
        side_json,
        row.get("Image URL", ""),
        row.get("Manufacturer", ""),
        row.get("Excellent Review %", 0),
        row.get("Average Review", 0),
        row.get("Poor Review %", 0)
    ))

conn.commit()
conn.close()

print(f"✅ Database successfully created from file {CSV_PATH}!")
print(f"   Number of medicines: {len(df)}")