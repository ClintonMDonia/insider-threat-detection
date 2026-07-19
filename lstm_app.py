import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title("Insider Threat Detection System")
st.subheader("LSTM Deep Learning Model — UNIZIK Cybersecurity & Digital Forensics")
st.markdown("---")

model = load_model("lstm_model.keras")

features = [
    "total_logons", "after_hours_logons", "weekend_logons",
    "unique_pcs", "usb_connects", "usb_disconnects",
    "total_files_accessed", "file_opens", "file_writes",
    "file_copies", "files_to_removable"
]

SEQUENCE_LENGTH = 7

uploaded_file = st.file_uploader("Upload User Activity CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    if all(f in df.columns for f in features):
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        results = []
        for user, group in df.groupby("user"):
            group = group.reset_index(drop=True)
            if len(group) >= SEQUENCE_LENGTH:
                for i in range(len(group) - SEQUENCE_LENGTH):
                    seq = group[features].iloc[i:i+SEQUENCE_LENGTH].values
                    seq = seq.reshape(1, SEQUENCE_LENGTH, len(features))
                    prob = model.predict(seq, verbose=0)[0][0]
                    label = "🔴 Suspicious" if prob > 0.5 else "🟢 Normal"
                    results.append({
                        "user": user,
                        "date": group["date"].iloc[i+SEQUENCE_LENGTH] if "date" in group.columns else i,
                        "risk_score": round(float(prob), 4),
                        "status": label
                    })

        results_df = pd.DataFrame(results)

        st.markdown("### Detection Results")
        st.dataframe(results_df)

        suspicious_count = (results_df["status"].str.contains("Suspicious")).sum()
        normal_count = (results_df["status"].str.contains("Normal")).sum()

        st.markdown("### Summary")
        col1, col2 = st.columns(2)
        col1.metric("Suspicious Records", suspicious_count)
        col2.metric("Normal Records", normal_count)

        fig, ax = plt.subplots()
        ax.bar(["Normal", "Suspicious"], [normal_count, suspicious_count],
               color=["green", "red"])
        ax.set_title("Detection Summary")
        st.pyplot(fig)
    else:
        st.error("CSV is missing required feature columns.")
