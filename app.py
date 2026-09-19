import os
import tempfile

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HelmetGuard AI",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
        .stApp {
            background: #0b1120;
        }

        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #1f2937;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        .hero {
            padding: 30px;
            border-radius: 22px;
            margin-bottom: 25px;
            background: linear-gradient(135deg, #172554, #1e3a8a, #312e81);
            border: 1px solid #3730a3;
        }

        .hero h1 {
            color: white;
            font-size: 42px;
            margin: 0;
        }

        .hero p {
            color: #c7d2fe;
            font-size: 17px;
            margin-top: 8px;
        }

        .card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 18px;
            padding: 20px;
            text-align: center;
        }

        .number {
            color: #60a5fa;
            font-size: 30px;
            font-weight: 800;
        }

        .label {
            color: #94a3b8;
            font-size: 14px;
        }

        .success-box {
            background: #064e3b;
            border: 1px solid #10b981;
            color: #a7f3d0;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            font-weight: 700;
        }

        .warning-box {
            background: #450a0a;
            border: 1px solid #ef4444;
            color: #fecaca;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            font-weight: 700;
        }

        .info-box {
            background: #172554;
            border: 1px solid #2563eb;
            color: #bfdbfe;
            padding: 16px;
            border-radius: 12px;
        }

        .stButton > button {
            width: 100%;
            border: 0;
            border-radius: 10px;
            padding: 12px;
            font-weight: 700;
            color: white;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
        }

        .stButton > button:hover {
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.35);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return YOLO("model/best.pt")


model = load_model()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🪖 HelmetGuard AI")
    st.caption("AI-powered helmet compliance detection")

    st.divider()

    st.markdown("### ⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    st.divider()

    st.markdown("### 🤖 Model")
    st.write("**YOLO11n**")
    st.write("Custom trained helmet detector")

    st.divider()

    st.markdown("### 🎯 Classes")
    st.write("🟢 With Helmet")
    st.write("🔴 Without Helmet")


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>🪖 HelmetGuard AI</h1>
        <p>
            Intelligent computer vision for helmet detection
            and road-safety monitoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Helmet Detection Dashboard")
st.caption("Upload an image or video and analyze it using your trained YOLO model.")


# =========================================================
# DASHBOARD METRICS
# =========================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        '<div class="card"><div class="number">YOLO11</div>'
        '<div class="label">Detection Model</div></div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        '<div class="card"><div class="number">2</div>'
        '<div class="label">Detection Classes</div></div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        '<div class="card"><div class="number">640</div>'
        '<div class="label">Image Size</div></div>',
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        '<div class="card"><div class="number">AI</div>'
        '<div class="label">Powered Detection</div></div>',
        unsafe_allow_html=True,
    )


st.markdown("")


# =========================================================
# TABS
# =========================================================
image_tab, video_tab = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])


# =========================================================
# IMAGE DETECTION
# =========================================================
with image_tab:
    st.subheader("Upload an Image")

    uploaded_image = st.file_uploader(
        "Choose a traffic or rider image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        left, right = st.columns(2)

        with left:
            st.markdown("### Original")
            st.image(image, use_container_width=True)

        if st.button("🚀 Detect Helmets", key="detect_image"):
            with st.spinner("AI is analyzing the image..."):
                results = model.predict(
                    source=image,
                    conf=confidence,
                    verbose=False,
                )

            result = results[0]
            annotated = result.plot()

            with right:
                st.markdown("### AI Detection")
                st.image(annotated, use_container_width=True)

            helmet_count = 0
            no_helmet_count = 0

            if result.boxes is not None:
                for cls in result.boxes.cls:
                    class_id = int(cls)

                    if class_id == 0:
                        helmet_count += 1
                    elif class_id == 1:
                        no_helmet_count += 1

            total = helmet_count + no_helmet_count

            st.markdown("### 📊 Detection Summary")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Total Detected", total)

            with m2:
                st.metric("🟢 With Helmet", helmet_count)

            with m3:
                st.metric("🔴 Without Helmet", no_helmet_count)

            if no_helmet_count > 0:
                st.markdown(
                    f"""
                    <div class="warning-box">
                        ⚠️ {no_helmet_count} rider(s) detected without a helmet
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif helmet_count > 0:
                st.markdown(
                    """
                    <div class="success-box">
                        ✅ All detected riders are wearing helmets
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="info-box">
                        ℹ️ No helmet-related objects were detected.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =========================================================
# VIDEO DETECTION
# =========================================================
with video_tab:
    st.subheader("Upload a Traffic Video")

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("🚀 Analyze Video", key="detect_video"):
            input_path = None
            output_path = None

            try:
                with st.spinner(
                    "AI is processing the video. This may take some time..."
                ):
                    input_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4",
                    )
                    input_file.write(uploaded_video.getbuffer())
                    input_file.close()
                    input_path = input_file.name

                    results = model.predict(
                        source=input_path,
                        conf=confidence,
                        save=True,
                        verbose=False,
                    )

                    save_dir = results[0].save_dir
                    output_candidates = []

                    for filename in os.listdir(save_dir):
                        if filename.lower().endswith(
                            (".mp4", ".avi", ".mov", ".mkv")
                        ):
                            output_candidates.append(
                                os.path.join(save_dir, filename)
                            )

                    if output_candidates:
                        output_path = output_candidates[0]

                        st.success("Video processing completed!")
                        st.markdown("### 🎬 Detection Result")
                        st.video(output_path)

                        with open(output_path, "rb") as video_file:
                            st.download_button(
                                "⬇️ Download Detection Video",
                                data=video_file.read(),
                                file_name="helmet_detection_result.mp4",
                                mime="video/mp4",
                            )
                    else:
                        st.warning(
                            "Detection completed, but the processed video "
                            "file could not be located automatically."
                        )

            except Exception as e:
                st.error(f"Video processing failed: {e}")

            finally:
                if input_path and os.path.exists(input_path):
                    os.remove(input_path)


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:18px;">
        🪖 <b>HelmetGuard AI</b><br>
        YOLO11 • Ultralytics • Streamlit<br>
        <small>AI-powered helmet detection system</small>
    </div>
    """,
    unsafe_allow_html=True,
)
