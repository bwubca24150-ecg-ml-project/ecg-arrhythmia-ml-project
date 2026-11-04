"""
app.py
------
Streamlit web interface for the ECG arrhythmia classifier.

Run with:  streamlit run app.py
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_loader import CLASS_DESCRIPTIONS, CLASSES, PROJECT_ROOT, find_records
from src.model import load_model, predict_beats
from src.preprocessing import BEAT_LEN, PRE, RR_COLUMNS, build_record_dataset
from predict import SIGNAL_COLUMNS, read_beats_csv

st.set_page_config(page_title="ECG Arrhythmia Classifier", page_icon="🫀", layout="wide")

st.title("🫀 ECG Arrhythmia Classification")
st.caption("Machine Learning-Based ECG Signal Classification for Arrhythmia Detection")
st.warning(
    "Academic prototype only. This is NOT a clinically validated diagnostic system "
    "and must not be used for medical decisions."
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_records():
    return {k: str(v) for k, v in find_records(exclude_paced=False).items()}


@st.cache_data
def get_record_data(path):
    return build_record_dataset(path)


try:
    bundle = get_model()
except FileNotFoundError:
    st.error("Trained model not found. Run `python train.py` first, then restart this app.")
    st.stop()


def show_results(beats, result_df):
    """Common result display for both modes."""
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Predicted class counts")
        counts = result_df["predicted_class"].value_counts().reindex(CLASSES).fillna(0).astype(int)
        st.bar_chart(counts)
    with c2:
        st.subheader("Class legend")
        st.table(pd.DataFrame({"Class": CLASSES, "Description": [CLASS_DESCRIPTIONS[c] for c in CLASSES]}))

    st.subheader("Beat-by-beat results")
    st.dataframe(result_df, height=300)
    st.download_button(
        "Download results as CSV",
        result_df.to_csv(index=False).encode(),
        file_name="predictions.csv",
        mime="text/csv",
    )

    st.subheader("View a single beat")
    idx = st.slider("Beat number", 0, len(beats) - 1, 0)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(beats[idx], color="#2b6cb0")
    ax.axvline(PRE, color="red", linestyle="--", alpha=0.5, label="R peak")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(f"Beat {idx} - predicted: {result_df['predicted_class'].iloc[idx]}")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


mode = st.sidebar.radio("Input source", ["Dataset record (MIT-BIH)", "Upload beats CSV"])

if mode == "Dataset record (MIT-BIH)":
    records = get_records()
    if not records:
        st.error("No MIT-BIH records found in the data/ folder.")
    else:
        name = st.sidebar.selectbox("Record", list(records.keys()))
        max_beats = st.sidebar.slider("Number of beats", 10, 500, 100, step=10)
        beats, rr, labels = get_record_data(records[name])
        beats, rr, labels = beats[:max_beats], rr[:max_beats], labels[:max_beats]
        pred, proba = predict_beats(bundle, beats, rr)

        result_df = pd.DataFrame(
            {
                "beat": range(len(pred)),
                "true_label": labels,
                "predicted_class": pred,
                "description": [CLASS_DESCRIPTIONS[p] for p in pred],
                "confidence": proba.max(axis=1).round(3),
            }
        )
        agree = (result_df["true_label"] == result_df["predicted_class"]).mean()
        st.metric("Agreement with dataset annotations", f"{agree:.1%}")
        st.caption("If this record was used for training, the agreement is optimistic.")
        show_results(beats, result_df)

else:
    st.sidebar.markdown(
        f"CSV needs columns **s0 ... s{BEAT_LEN - 1}** (one beat per row). "
        f"Optional: {', '.join(RR_COLUMNS)} (seconds)."
    )
    up = st.sidebar.file_uploader("Upload beats CSV", type="csv")
    if up is None:
        st.info("Upload a CSV file from the sidebar. Create a sample with `python predict.py --make_sample`.")
    else:
        try:
            df, beats, rr = read_beats_csv(up)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
        pred, proba = predict_beats(bundle, beats, rr)
        result_df = pd.DataFrame(
            {
                "beat": range(len(pred)),
                "predicted_class": pred,
                "description": [CLASS_DESCRIPTIONS[p] for p in pred],
                "confidence": proba.max(axis=1).round(3),
            }
        )
        if "true_label" in df.columns:
            result_df.insert(1, "true_label", df["true_label"].values)
        show_results(beats, result_df)

with st.expander("Model performance (from last training run)"):
    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        st.write(f"Test accuracy: **{m['accuracy']:.4f}** on {m['n_test_beats']} beats "
                 f"(records: {', '.join(m['test_records'])})")
        st.text((PROJECT_ROOT / "reports" / "classification_report.txt").read_text())
        cm_img = PROJECT_ROOT / "reports" / "confusion_matrix.png"
        if cm_img.exists():
            st.image(str(cm_img))
    else:
        st.write("No metrics found. Run `python train.py`.")
