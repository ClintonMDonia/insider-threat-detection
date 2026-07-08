
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Insider Threat Detection System")
st.subheader("UNIZIK — Cybersecurity & Digital Forensics")
st.markdown("---")

model = joblib.load("rf_model.pkl")

uploaded_file = st.file_uploader("Upload a user activity CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    features = [
        "total_logons", "after_hours_logons", "weekend_logons",
        "unique_pcs", "usb_connects", "usb_disconnects",
        "total_files_accessed", "file_opens", "file_writes",
        "file_copies", "files_to_removable"
    ]

    if all(f in df.columns for f in features):
        predictions = model.predict(df[features])
        df["prediction"] = predictions
        df["status"] = df["prediction"].apply(
            lambda x: "🔴 Suspicious" if x == 1 else "🟢 Normal"
        )

        st.markdown("### Detection Results")
        st.dataframe(df[["user", "date", "status"]])

        suspicious_count = (predictions == 1).sum()
        normal_count = (predictions == 0).sum()

        st.markdown("### Summary")
        col1, col2 = st.columns(2)
        col1.metric("Suspicious Users", suspicious_count)
        col2.metric("Normal Users", normal_count)

        fig, ax = plt.subplots()
        ax.bar(["Normal", "Suspicious"], [normal_count, suspicious_count],
               color=["green", "red"])
        ax.set_title("Detection Summary")
        st.pyplot(fig)
    else:
        st.error("CSV is missing required columns.")
