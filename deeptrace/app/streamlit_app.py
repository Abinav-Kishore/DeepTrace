"""Streamlit front-end: upload a video, see verdict + per-frame timeline.

Run with:
    streamlit run deeptrace/app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="DeepTrace (research prototype)", page_icon="🎬", layout="centered")

st.title("🎬 DeepTrace — Deepfake Detection (research prototype)")

st.warning(
    "**Disclaimer:** This is a research prototype trained on a small subset of the "
    "DFDC dataset. It is **not** a forensic tool and should not be used to make "
    "real-world determinations about media authenticity."
)

with st.sidebar:
    st.subheader("Model")
    kind = st.radio("Choose model", ["temporal", "baseline"], index=0)
    st.caption(
        "temporal = frozen ResNet18 + BiGRU; baseline = frame-averaged ResNet18. "
        "Both are trained on the DFDC sample subset (see README for metrics)."
    )


def run_impl(upload_bytes: bytes, suffix: str, kind: str) -> dict:
    import tempfile

    from deeptrace.inference import predict_video

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload_bytes)
        path = Path(tmp.name)
    try:
        return predict_video(path, kind=kind)
    finally:
        path.unlink(missing_ok=True)


upload = st.file_uploader(
    "Upload a short video (mp4/mov/avi/webm/mkv, <100MB)",
    type=["mp4", "mov", "avi", "webm", "mkv"],
)

if upload is None:
    st.info("Upload a video to get a prediction.")
    st.stop()

if upload.size > 100 * 1024 * 1024:
    st.error("File is too large (>100 MB). Please upload a shorter clip.")
    st.stop()

with st.spinner("Extracting frames and running the model..."):
    try:
        result = run_impl(upload.getvalue(), Path(upload.name).suffix.lower(), kind)
    except ValueError as e:
        st.error(f"Could not process this video: {e}")
        st.stop()
    except RuntimeError as e:
        st.error(f"{e}")
        st.stop()

prob = result["video_probability"]
verdict = result["verdict"]

st.markdown(f"## Verdict: **{verdict}**")
st.progress(min(max(prob, 0.0), 1.0), text=f"fake probability: {prob:.3f}")
st.caption(
    f"model={result['model']} · frames used={result['frames_extracted']} · "
    f"faces detected in {result['faces_detected']}/{result['frames_extracted']} frames"
)

st.subheader("Per-frame fake-probability timeline")
st.bar_chart({"frame i": result["per_frame_scores"]})
st.caption(
    "Frames are evenly spaced across the video; each score is the model's "
    "fake probability for that frame/window. A repeated high tail suggests "
    "consistent manipulation signals across the video."
)

st.warning("Research prototype — not a forensic tool. Do not rely on this in production.")
