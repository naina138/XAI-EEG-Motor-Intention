import streamlit as st
import numpy as np
import mne
import joblib
import shap
import matplotlib.pyplot as plt
import tempfile

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="EEG Classifier", layout="wide")

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("eeg_model.pkl")

label_map = {
    1: "Rest",
    2: "Left Hand",
    3: "Right Hand"
}

# -----------------------------
# Feature Names
# -----------------------------
feature_names = (
    [f"Ch{i}_mean" for i in range(64)] +
    [f"Ch{i}_var" for i in range(64)] +
    [f"Ch{i}_std" for i in range(64)]
)

# -----------------------------
# UI
# -----------------------------
st.title("🧠 EEG Motor Intention Classifier")
st.markdown("### Explainable AI for Brain Signal Analysis")

uploaded_file = st.file_uploader("Upload EEG (.edf)", type=["edf"])

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if uploaded_file is not None:

    st.success("✅ File uploaded successfully")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Processing EEG data..."):

        # Load EEG
        raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)

        # Filter
        raw.filter(1., 40., verbose=False)

        # Extract events
        events, event_id = mne.events_from_annotations(raw)

        if len(events) == 0:
            st.error("❌ No events found in file")

        else:
            # Epoching
            epochs = mne.Epochs(
                raw,
                events,
                event_id,
                tmin=0,
                tmax=2,
                baseline=None,
                preload=True,
                verbose=False
            )

            data = epochs.get_data()

            # -----------------------------
            # Feature extraction
            # -----------------------------
            X_mean = np.mean(data, axis=2)
            X_var = np.var(data, axis=2)
            X_std = np.std(data, axis=2)

            X = np.concatenate((X_mean, X_var, X_std), axis=1)

            # -----------------------------
            # Prediction
            # -----------------------------
            preds = model.predict(X)
            probs = model.predict_proba(X)

            preds_text = [label_map.get(p, str(p)) for p in preds]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Predictions")
                st.write(preds_text)

            with col2:
                st.subheader("📈 Confidence Scores")
                st.write(probs)

            # -----------------------------
            # Distribution
            # -----------------------------
            st.subheader("📊 Prediction Distribution")
            unique, counts = np.unique(preds_text, return_counts=True)
            st.bar_chart(dict(zip(unique, counts)))

            # -----------------------------
            # SHAP Explainability (FINAL CLEAN)
            # -----------------------------
            st.subheader("🧠 Explainability (SHAP)")

            if X.shape[0] < 10:
                st.warning("⚠️ Small sample size — explanation may be unstable")

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

                # 👉 Get most frequent predicted class
                main_class = int(np.bincount(preds).argmax())

                st.markdown(
                    f"### 🔍 Explanation for Predicted Class: {label_map.get(main_class)}"
                )

                sv = shap_values[main_class - 1]

                # Fix shape if needed
                if sv.shape[0] != X.shape[0]:
                    sv = sv.T

                if sv.shape[1] != X.shape[1]:
                    st.warning("Skipping SHAP plot due to mismatch")
                else:
                    fig = plt.figure(figsize=(4, 4))

                    shap.summary_plot(
                        sv,
                        X,
                        feature_names=feature_names,
                        plot_type="bar",   # cleaner for small data
                        max_display=12,    # Limit features shown so plot doesn't get huge
                        plot_size=(8, 4),  # Control explicit size
                        show=False
                    )

                    st.pyplot(fig)

            except Exception as e:
                st.error(f"SHAP Error: {e}")

            # -----------------------------
            # Key Insight
            # -----------------------------
            st.subheader("🧩 Key Insight")

            try:
                if sv.shape[0] != X.shape[0]:
                    sv = sv.T

                importance = np.mean(np.abs(sv), axis=0)
                top_idx = np.argsort(importance)[-10:]

                top_features = [feature_names[i] for i in top_idx]

                st.write("Top contributing EEG features:")
                st.write(top_features)

            except:
                st.info("Not enough data for stable feature importance")

            st.success("✅ Analysis Complete!")