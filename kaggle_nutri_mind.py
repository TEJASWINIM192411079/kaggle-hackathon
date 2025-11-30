# ===============================================================
# SECTION 1 — IMPORTS
# ===============================================================
import os
import time
import pandas as pd
import google.generativeai as genai
import json

# ===============================================================
# LOAD API KEY FROM KAGGLE SECRETS (CORRECT METHOD)
# ===============================================================
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

api_key = user_secrets.get_secret("GOOGLE_API_KEY")  # MUST match secret name

if api_key is None or len(api_key) < 20:
    raise ValueError("❌ GOOGLE_API_KEY missing or invalid. Fix Kaggle Secrets.")

print("API Key Loaded Successfully ✔")

import google.generativeai as genai
genai.configure(api_key=api_key)



# ===============================================================
# SECTION 3 — LOAD MODEL
# ===============================================================
model = genai.GenerativeModel("gemini-2.5-flash-lite")
print("Model Loaded Successfully ✔")


# ===============================================================
# SECTION 4 — LOAD DATASET
# ===============================================================
df = pd.read_csv("/kaggle/input/hackathon/nutrients_csvfile.csv")
print("Dataset Loaded Successfully ✔")
print(df.head())


# ===============================================================
# SECTION 5 — PROCESS IN BATCHES (IMPROVED)
# ===============================================================

BATCH_SIZE = 25  # you can change this if needed
results = []

model = model  # keep the same model reference

for start in range(0, len(df), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(df))
    batch = df.iloc[start:end]

    print(f"Processing batch {start}–{end-1}...")

    # Prepare list of food entries
    batch_items = []
    for _, row in batch.iterrows():
        batch_items.append({
            "Food Name": row.get("Food Name", "Unknown Food"),
            "Calories":  row.get("Calories", "NA"),
            "Protein":   row.get("Protein",  "NA"),
            "Carbs":     row.get("Carbs",    "NA"),
            "Fat":       row.get("Fat",      "NA")
        })

    # Create ONE prompt per batch
    prompt = f"""
    You are a nutrition expert.
    Analyze the following list of foods.

    For EACH item return JSON in this format:
    {{
        "Food Name": "<name>",
        "Advice": "<2–3 line health suggestion>"
    }}

    Here is the list:
    {json.dumps(batch_items, indent=2)}
    """

    # Retry logic for API stability
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            text = response.text

            # Attempt to extract JSON list safely
            try:
                parsed = json.loads(text)
            except:
                # If model returns markdown or comments, extract JSON manually
                json_str = text[text.find("["):text.rfind("]") + 1]
                parsed = json.loads(json_str)

            # Append to results in SAME order
            for item in parsed:
                results.append([item["Food Name"], item["Advice"]])

            break  # success → exit retry loop

        except Exception as e:
            print(f"⚠ Error in batch {start}-{end-1}, attempt {attempt+1}: {e}")
            time.sleep(2)

    time.sleep(1)  # small cooldown to avoid any rate limits

print("Section 5 COMPLETED ✔ — All batches processed!")