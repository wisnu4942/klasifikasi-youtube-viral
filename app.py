import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import re
from datetime import datetime
import requests

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Klasifikasi Viralitas Konten YouTube",
    page_icon="🎬",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
}

.stMetric {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e9ecef;
}

.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.title-center {
    text-align: center;
    margin-bottom: 10px;
}

.subtitle-center {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.stNumberInput,
.stTextArea,
.stSelectbox {
    max-width: 500px;
}

textarea {
    border-radius: 12px !important;
}

div[data-baseweb="select"] {
    border-radius: 12px !important;
}
            
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    return joblib.load("lightgbm_moderate.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/000000/youtube-play.png",
        width=80
    )

    st.title("YouTube Viral Classifier")

    st.markdown("---")

    st.markdown("""
### 📌 Informasi Model

- **Algoritma:** LightGBM  
- **Teknik Data:** SMOTE 0.1  
- **Akurasi:** 96.81%  
- **F1-Score:** 82.35%  
- **AUC:** 98.69%
""")

    st.markdown("---")

    st.markdown("""
### 📊 Fitur Input

- Views
- Likes
- Comments
- Subscriber Channel
- Durasi Video
- Panjang Judul
- Jumlah Hashtag
- Jam Upload
- Hari Upload
""")

    st.markdown("---")
    st.caption("Skripsi - Klasifikasi Viralitas Konten YouTube")

# ============================================
# HEADER
# ============================================
st.markdown("""
<h1 class='title-center'>
🎬 Klasifikasi Viralitas Konten YouTube
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p class='subtitle-center'>
Sistem klasifikasi untuk menentukan apakah sebuah video termasuk kategori
<b>Viral</b> atau <b>Tidak Viral</b>
</p>
""", unsafe_allow_html=True)

# ============================================
# TABS
# ============================================
tab1, tab2 = st.tabs([
    "📝 Input Manual",
    "🔗 Input Link YouTube"
])

# ============================================
# TAB INPUT MANUAL
# ============================================
with tab1:

    st.markdown("## 📝 Form Data Video")

    with st.container(border=True):

        col1, col2 = st.columns(2, gap="large")

        # =========================
        # KOLOM KIRI
        # =========================
        with col1:

            st.markdown("### 📊 Statistik Video")

            views = st.number_input(
                "Views",
                min_value=0,
                value=100000,
                step=10000
            )

            likes = st.number_input(
                "Likes",
                min_value=0,
                value=5000,
                step=500
            )

            comments = st.number_input(
                "Comments",
                min_value=0,
                value=500,
                step=50
            )

            subscriber_count = st.number_input(
                "Subscriber Channel",
                min_value=0,
                value=500000,
                step=50000
            )

            duration_minutes = st.number_input(
                "Durasi Video (Menit)",
                min_value=0.5,
                value=5.0,
                step=0.5
            )

        # =========================
        # KOLOM KANAN
        # =========================
        with col2:

            st.markdown("### ⚙️ Detail Konten")

            title = st.text_area(
                "Judul Video",
                placeholder="Contoh: Kisah Horor Tengah Malam...",
                height=120
            )

            description = st.text_area(
                "Deskripsi Video",
                placeholder="Tambahkan hashtag jika ada...",
                height=120
            )

            col_time1, col_time2 = st.columns(2)

            with col_time1:
                publish_hour = st.selectbox(
                    "Jam Upload",
                    list(range(0, 24)),
                    index=18
                )

            with col_time2:
                publish_day_name = st.selectbox(
                    "Hari Upload",
                    [
                        "Senin",
                        "Selasa",
                        "Rabu",
                        "Kamis",
                        "Jumat",
                        "Sabtu",
                        "Minggu"
                    ]
                )

    # =========================
    # PREPROCESS
    # =========================
    duration_seconds = int(duration_minutes * 60)
    title_length = len(title)
    hashtags_count = description.count('#')

    day_map = {
        "Senin": 0,
        "Selasa": 1,
        "Rabu": 2,
        "Kamis": 3,
        "Jumat": 4,
        "Sabtu": 5,
        "Minggu": 6
    }

    publish_day = day_map[publish_day_name]

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # BUTTON
    # =========================
    if st.button(
        "🎯 Klasifikasi Sekarang",
        use_container_width=True,
        type="primary"
    ):

        input_data = np.array([[
            views,
            likes,
            comments,
            subscriber_count,
            duration_seconds,
            title_length,
            hashtags_count,
            publish_hour,
            publish_day
        ]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        st.markdown("## 📊 Hasil Klasifikasi")

        col1, col2, col3 = st.columns(3)

        with col1:

            if prediction == 1:
                st.success("🔥 VIRAL")
            else:
                st.error("📉 TIDAK VIRAL")

        with col2:
            st.metric(
                "Tingkat Keyakinan",
                f"{probability*100:.1f}%"
            )

        with col3:
            st.metric(
                "Model",
                "LightGBM"
            )

        st.progress(float(probability))

        if prediction == 1:
            st.success(
                "Video memiliki karakteristik yang mendekati kategori viral."
            )
        else:
            st.info(
                "Video memiliki karakteristik yang cenderung tidak viral."
            )

# ============================================
# TAB LINK YOUTUBE
# ============================================
with tab2:

    if YOUTUBE_API_KEY is None:

        st.error("❌ YouTube API Key tidak ditemukan")

    else:

        st.markdown("## 🔗 Input Link YouTube")

        with st.container(border=True):

            video_url = st.text_input(
                "Link Video YouTube",
                placeholder="https://youtube.com/watch?v=xxxx"
            )

            st.caption(
                "Sistem akan mengambil data video secara otomatis dari YouTube"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "🔍 Ambil Data & Klasifikasi",
            type="primary",
            use_container_width=True
        ):

            if not video_url:

                st.warning("Masukkan link video YouTube terlebih dahulu")

            else:

                with st.spinner("Mengambil data video dari YouTube..."):

                    video_id = extract_video_id(video_url)

                    if not video_id:

                        st.error("Link YouTube tidak valid")

                    else:

                        video_data, error = get_video_data(
                            video_id,
                            YOUTUBE_API_KEY
                        )

                        if error:

                            st.error(error)

                        else:

                            # =================================
                            # PREVIEW VIDEO
                            # =================================
                            st.markdown("## 📺 Preview Video")

                            preview1, preview2 = st.columns([1, 2])

                            with preview1:

                                st.image(
                                    video_data['thumbnail'],
                                    use_container_width=True
                                )

                            with preview2:

                                st.markdown(
                                    f"### {video_data['title']}"
                                )

                                st.write(
                                    f"📺 Channel: {video_data['channel_name']}"
                                )

                                st.link_button(
                                    "Buka di YouTube",
                                    video_data['video_url']
                                )

                            # =================================
                            # DETAIL DATA
                            # =================================
                            with st.expander(
                                "📊 Detail Data Video",
                                expanded=False
                            ):

                                d1, d2, d3 = st.columns(3)

                                with d1:
                                    st.metric(
                                        "Views",
                                        f"{video_data['views']:,}"
                                    )

                                    st.metric(
                                        "Likes",
                                        f"{video_data['likes']:,}"
                                    )

                                with d2:
                                    st.metric(
                                        "Comments",
                                        f"{video_data['comments']:,}"
                                    )

                                    st.metric(
                                        "Subscriber",
                                        f"{video_data['subscriber_count']:,}"
                                    )

                                with d3:
                                    st.metric(
                                        "Durasi",
                                        f"{video_data['duration_seconds']} detik"
                                    )

                                    st.metric(
                                        "Hashtag",
                                        video_data['hashtags_count']
                                    )

                            # =================================
                            # KLASIFIKASI
                            # =================================
                            input_data = np.array([[
                                video_data['views'],
                                video_data['likes'],
                                video_data['comments'],
                                video_data['subscriber_count'],
                                video_data['duration_seconds'],
                                video_data['title_length'],
                                video_data['hashtags_count'],
                                video_data['publish_hour'],
                                video_data['publish_day']
                            ]])

                            input_scaled = scaler.transform(input_data)

                            prediction = model.predict(
                                input_scaled
                            )[0]

                            probability = model.predict_proba(
                                input_scaled
                            )[0][1]

                            st.markdown("---")
                            st.markdown("## 🎯 Hasil Klasifikasi")

                            r1, r2, r3 = st.columns(3)

                            with r1:

                                if prediction == 1:
                                    st.success("🔥 VIRAL")
                                else:
                                    st.error("📉 TIDAK VIRAL")

                            with r2:

                                st.metric(
                                    "Tingkat Keyakinan",
                                    f"{probability*100:.1f}%"
                                )

                            with r3:

                                st.metric(
                                    "Model",
                                    "LightGBM"
                                )

                            st.progress(float(probability))

                            if prediction == 1:

                                st.success(
                                    "Video memiliki karakteristik viral."
                                )

                            else:

                                st.info(
                                    "Video memiliki karakteristik tidak viral."
                                )

# ============================================
# FOOTER
# ============================================
st.markdown("---")

st.markdown("""
<div style='text-align:center; color:gray; padding:10px;'>
    <p>
        YouTube Viral Classifier • LightGBM • Streamlit
    </p>
</div>
""", unsafe_allow_html=True)