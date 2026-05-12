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
    page_title="YouTube Viral Predictor",
    page_icon="🎬",
    layout="wide"
)

# ============================================
# LOAD MODEL (Cached)
# ============================================
@st.cache_resource
def load_model():
    return joblib.load("lightgbm_moderate.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_data
def load_feature_info():
    with open("feature_info.json", "r") as f:
        return json.load(f)

model = load_model()
scaler = load_scaler()
feature_info = load_feature_info()
feature_names = feature_info['feature_names']

# ============================================
# AMBIL API KEY DARI SECRETS
# ============================================
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
except:
    YOUTUBE_API_KEY = None
    st.warning("YouTube API Key tidak ditemukan. Fitur input link YouTube tidak tersedia.")

# ============================================
# FUNGSI BANTUAN
# ============================================
def extract_video_id(url):
    patterns = [
        r'(?:youtube\.com\/watch\?v=)([\w-]+)',
        r'(?:youtu\.be\/)([\w-]+)',
        r'(?:youtube\.com\/embed\/)([\w-]+)',
        r'(?:youtube\.com\/v\/)([\w-]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def parse_duration(duration_str):
    import re
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration_str)
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds

def get_subscriber_count(channel_id, api_key):
    url = f"https://www.googleapis.com/youtube/v3/channels?part=statistics&id={channel_id}&key={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get('items'):
            return int(data['items'][0]['statistics'].get('subscriberCount', 0))
        return 0
    except:
        return 0

def get_video_data(video_id, api_key):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics,contentDetails&id={video_id}&key={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if not data.get('items'):
            return None, "Video tidak ditemukan"
        
        item = data['items'][0]
        snippet = item['snippet']
        statistics = item['statistics']
        content_details = item['contentDetails']
        
        title = snippet.get('title', '')
        description = snippet.get('description', '')
        published_at = snippet.get('publishedAt', '')
        channel_id = snippet.get('channelId', '')
        
        views = int(statistics.get('viewCount', 0))
        likes = int(statistics.get('likeCount', 0))
        comments = int(statistics.get('commentCount', 0))
        
        duration_str = content_details.get('duration', 'PT0S')
        duration_seconds = parse_duration(duration_str)
        
        title_length = len(title)
        hashtags_count = description.count('#')
        
        pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        publish_hour = pub_date.hour
        publish_day = pub_date.weekday()
        
        subscriber_count = get_subscriber_count(channel_id, api_key)
        
        video_data = {
            'title': title,
            'views': views,
            'likes': likes,
            'comments': comments,
            'subscriber_count': subscriber_count,
            'duration_seconds': duration_seconds,
            'title_length': title_length,
            'hashtags_count': hashtags_count,
            'publish_hour': publish_hour,
            'publish_day': publish_day,
            'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
            'channel_name': snippet.get('channelTitle', ''),
            'video_url': f"https://youtube.com/watch?v={video_id}"
        }
        
        return video_data, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/youtube-play.png", width=80)
    st.title("YouTube Viral Predictor")
    st.markdown("---")
    st.markdown("""
    **Model:** LightGBM + SMOTE 0.1  
    **Akurasi:** 96.81%  
    **F1-Score:** 82.35%  
    **AUC:** 98.69%
    
    **Fitur yang digunakan:**
    - Views, Likes, Comments
    - Subscriber Count
    - Duration
    - Title Length
    - Hashtags Count
    - Publish Hour & Day
    """)
    st.markdown("---")
    st.caption("Skripsi - Klasifikasi Viralitas Konten YouTube Niche Horor Anak")

# ============================================
# MAIN CONTENT
# ============================================
st.title("🎬 YouTube Viral Predictor")
st.markdown("Prediksi apakah video YouTube Anda akan **VIRAL** atau **TIDAK VIRAL**")

tab1, tab2 = st.tabs(["📝 Input Manual", "🔗 Input Link YouTube"])

# ============================================
# TAB 1: INPUT MANUAL
# ============================================
with tab1:
    st.markdown("Masukkan data video secara manual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        views = st.number_input("Views", min_value=0, value=100000, step=10000, key="manual_views")
        likes = st.number_input("Likes", min_value=0, value=5000, step=500, key="manual_likes")
        comments = st.number_input("Comments", min_value=0, value=500, step=50, key="manual_comments")
        subscriber_count = st.number_input("Subscriber Channel", min_value=0, value=500000, step=50000, key="manual_sub")
    
    with col2:
        duration_minutes = st.number_input("Durasi (menit)", min_value=0.5, value=5.0, step=0.5, key="manual_duration")
        duration_seconds = int(duration_minutes * 60)
        title = st.text_input("Judul Video", placeholder="Contoh: LILLY LOVE BRAIDS BEACH DAY!!", key="manual_title")
        title_length = len(title)
        description = st.text_area("Deskripsi (opsional)", placeholder="Masukkan deskripsi... #hashtag", key="manual_desc")
        hashtags_count = description.count('#')
        publish_hour = st.selectbox("Jam Upload", list(range(0, 24)), index=18, key="manual_hour")
        publish_day_name = st.selectbox("Hari Upload", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"], key="manual_day")
        day_map = {"Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, "Jumat": 4, "Sabtu": 5, "Minggu": 6}
        publish_day = day_map[publish_day_name]
    
    if st.button("Prediksi Manual", type="primary"):
        input_data = np.array([[
            views, likes, comments, subscriber_count,
            duration_seconds, title_length, hashtags_count,
            publish_hour, publish_day
        ]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        st.subheader("Hasil Prediksi")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            if prediction == 1:
                st.metric("Status", "🔥 VIRAL", delta="Potensi Tinggi")
            else:
                st.metric("Status", "📉 TIDAK VIRAL", delta="Potensi Rendah")
        with col_r2:
            st.metric("Probabilitas", f"{probability*100:.1f}%")
        with col_r3:
            st.metric("Model", "LightGBM")
        st.progress(float(probability))

# ============================================
# TAB 2: INPUT LINK YOUTUBE
# ============================================
with tab2:
    if YOUTUBE_API_KEY is None:
        st.error("❌ YouTube API Key tidak ditemukan. Fitur ini tidak tersedia.")
        st.info("Admin telah mengkonfigurasi API Key di backend. Silakan coba lagi nanti.")
    else:
        st.markdown("Cukup paste link video YouTube, sistem akan mengambil data secara otomatis")
        
        video_url = st.text_input("Link Video YouTube", placeholder="https://youtube.com/watch?v=abc123", key="url_input")
        
        if st.button("🔍 Ambil Data & Prediksi", type="primary"):
            if not video_url:
                st.error("Harap masukkan link video YouTube")
            else:
                with st.spinner("Mengambil data dari YouTube..."):
                    video_id = extract_video_id(video_url)
                    if not video_id:
                        st.error("Link YouTube tidak valid")
                    else:
                        video_data, error = get_video_data(video_id, YOUTUBE_API_KEY)
                        
                        if error:
                            st.error(error)
                        else:
                            st.subheader("📺 Preview Video")
                            col_preview1, col_preview2 = st.columns([1, 2])
                            with col_preview1:
                                st.image(video_data['thumbnail'], width=200)
                            with col_preview2:
                                st.markdown(f"**Judul:** {video_data['title'][:100]}...")
                                st.markdown(f"**Channel:** {video_data['channel_name']}")
                                st.markdown(f"[Buka di YouTube]({video_data['video_url']})")
                            
                            with st.expander("📊 Data yang Diekstrak dari YouTube"):
                                st.write(f"Views: {video_data['views']:,}")
                                st.write(f"Likes: {video_data['likes']:,}")
                                st.write(f"Comments: {video_data['comments']:,}")
                                st.write(f"Subscriber Channel: {video_data['subscriber_count']:,}")
                                st.write(f"Durasi: {video_data['duration_seconds']} detik")
                                st.write(f"Panjang Judul: {video_data['title_length']} karakter")
                                st.write(f"Jumlah Hashtag: {video_data['hashtags_count']}")
                                st.write(f"Jam Upload: {video_data['publish_hour']}:00")
                                st.write(f"Hari Upload: {['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'][video_data['publish_day']]}")
                            
                            input_data = np.array([[
                                video_data['views'], video_data['likes'], video_data['comments'],
                                video_data['subscriber_count'], video_data['duration_seconds'],
                                video_data['title_length'], video_data['hashtags_count'],
                                video_data['publish_hour'], video_data['publish_day']
                            ]])
                            
                            input_scaled = scaler.transform(input_data)
                            prediction = model.predict(input_scaled)[0]
                            probability = model.predict_proba(input_scaled)[0][1]
                            
                            st.subheader("🎯 Hasil Prediksi")
                            col_r1, col_r2, col_r3 = st.columns(3)
                            with col_r1:
                                if prediction == 1:
                                    st.metric("Status", "🔥 VIRAL", delta="Potensi Tinggi")
                                else:
                                    st.metric("Status", "📉 TIDAK VIRAL", delta="Potensi Rendah")
                            with col_r2:
                                st.metric("Probabilitas", f"{probability*100:.1f}%")
                            with col_r3:
                                st.metric("Model", "LightGBM")
                            
                            st.progress(float(probability))
                            
                            if prediction == 1:
                                st.success("✅ Video ini berpotensi VIRAL! Perbanyak promosi di 6 jam pertama.")
                            else:
                                st.info("📌 Video ini kurang berpotensi viral. Coba upload di jam 18:00 atau perbaiki thumbnail.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>YouTube Viral Predictor | Skripsi - Klasifikasi Viralitas Konten YouTube Niche Horor Anak</p>
    <p>Model: LightGBM + SMOTE 0.1 | Akurasi: 96.81% | F1-Score: 82.35%</p>
</div>
""", unsafe_allow_html=True)