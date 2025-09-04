# app_agri_final_fixed.py
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os, uuid, io, base64
from datetime import datetime
import requests
from torchvision import transforms
from torchvision.models import mobilenet_v2
import torch

# ------------------- CONFIG -------------------
st.set_page_config(page_title="AgriConnect MRV", page_icon="🌾", layout="wide")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
SUB_CSV = os.path.join(DATA_DIR, "submissions.csv")

# ------------------- STORAGE -------------------
def init_storage():
    if not os.path.exists(SUB_CSV):
        pd.DataFrame(columns=[
            "submission_id","farmer_id","farmer_name","state","timestamp",
            "image_name","detected_crop","detected_health","notes","verified","verifier","verify_ts"
        ]).to_csv(SUB_CSV, index=False)
init_storage()

def load_subs(): return pd.read_csv(SUB_CSV)
def save_subs(df): df.to_csv(SUB_CSV, index=False)
def save_image_bytes(img_bytes, name=None):
    if not name:
        name = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(DATA_DIR, name)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return name

# ------------------- MODEL -------------------
@st.cache_resource
def load_model():
    model = mobilenet_v2(pretrained=True)
    model.eval()
    return model
model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

IMAGENET_LABELS = {0: "Wheat", 1: "Rice", 2: "Millet", 3:"Maize", 4:"Sugarcane"} # demo

def detect_image(image: Image.Image):
    image = image.resize((224,224))  # Resize for speed
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(img)
        pred = out.argmax(dim=1).item()
    crop = IMAGENET_LABELS.get(pred, "Unknown Plant")
    health = "Healthy"  # demo
    confidence = round(torch.softmax(out, dim=1)[0][pred].item(),2)
    return {"crop":crop,"health":health,"confidence":confidence}

# ------------------- WEATHER & CROP SUGGESTION -------------------
STATE_COORDS = {
    "Andhra Pradesh": (15.9129,79.74),
    "Bihar": (25.0961,85.3131),
    "Karnataka": (15.3173,75.7139),
    "Maharashtra": (19.7515,75.7139),
    "Tamil Nadu": (11.1271,78.6569),
    "Uttar Pradesh": (26.8467,80.9462),
    "Other": (20.5937,78.9629)
}

def get_weather_data(latitude, longitude):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        response = requests.get(url)
        data = response.json()
        return data
    except:
        return None

def suggest_crop(weather_data):
    try:
        temp = weather_data['current_weather'].get('temperature', 0)
        rainfall = weather_data['current_weather'].get('precipitation', 0)

        if rainfall > 200 and 25 <= temp <= 35:
            return "Rice"
        elif 10 <= temp <= 20:
            return "Wheat"
        elif 20 <= temp <= 30 and rainfall < 150:
            return "Maize"
        else:
            return "Crop data unavailable for current conditions"
    except:
        return "Weather data unavailable"

# ------------------- CSS / THEME -------------------
st.markdown("""
<style>
* { color: #008c00 !important; font-weight: 700 !important; }
.stTextInput input, .stTextArea textarea, .stSelectbox select, .stFileUploader input {
    color: #ffffff !important; 
    background-color: #0B3D0B !important;
    font-weight:700 !important;
    border-radius:8px !important;
    padding: 8px !important;
    box-shadow: 0 3px 6px rgba(0,0,0,0.16) !important;
}
input::placeholder, textarea::placeholder { color: #dddddd !important; opacity:1 !important; }
.stButton>button {
    background: linear-gradient(90deg,#1a3b1a,#2e6b2e) !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    border-radius:12px !important;
    padding: 10px 25px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25) !important;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.35) !important;
}
.sidebar * { color: #0B3D0B !important; font-weight:700 !important; }
.stApp { background: linear-gradient(180deg,#fbf9ef,#f2f7f0); }
.card { background: #fff; border-radius:12px; padding:14px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); margin-bottom:12px; }
.title { font-size:28px; font-weight:900; color:#0B3D0B; }
.small-muted { color:#0B3D0B; font-size:15px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ------------------- SESSION -------------------
if "user" not in st.session_state: st.session_state.user = None
if "uploaded_image" not in st.session_state: st.session_state.uploaded_image = None
if "analyze_result" not in st.session_state: st.session_state.analyze_result = None
if "notification" not in st.session_state: st.session_state.notification = []

# ------------------- STATES -------------------
INDIAN_STATES = ["Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana",
                 "Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur",
                 "Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu",
                 "Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal","Other"]

# ------------------- LANDING PAGE -------------------
if st.session_state.user is None:
    st.markdown("<div class='title'>🌾 AgriConnect MRV</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Farmer-friendly MRV — AI detection, crop suggestions, modern techniques & notifications</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    role = st.selectbox("I am a", ["Farmer","Official"])
    name = st.text_input("Name")
    
    if role=="Farmer":
        state = st.selectbox("State", INDIAN_STATES)
        lang = st.radio("Language", ["English","हिंदी"], horizontal=True)
        password = None
    else:
        password = st.text_input("Password", type="password")
        state = None
        lang = "en"
    
    if st.button("Sign in"):
        if not name:
            st.error("Please enter your name")
        else:
            uid = (("F" if role=="Farmer" else "O") + "-" + base64.urlsafe_b64encode(name.encode()).decode()[:8])
            st.session_state.user = {"role":role,"name":name,"id":uid,"state":state,"lang":("hi" if lang.startswith("ह") else "en")}
            st.success(f"Signed in as {name} ({uid})")
            st.rerun()

# ------------------- MAIN APP -------------------
else:
    user = st.session_state.user
    sidebar, main = st.columns([1,4])
    
    with sidebar:
        st.markdown(f"**{user['name']}** ({user['id']})")
        st.markdown(f"Role: {user['role']}")
        st.markdown("---")
        if user['role']=="Farmer":
            page = st.radio("Menu", ["Upload Image","My Submissions","Notifications","Crop Info","Government Schemes","Logout"], index=0)
        else:
            page = st.radio("Menu", ["Pending Verifications","Verified Records","Logout"], index=0)
        st.markdown("---")
    
    with main:
        df = load_subs()
        # ------------------- FARMER -------------------
        if user['role']=="Farmer":
            if page=="Logout":
                st.session_state.user = None
                st.rerun()
            
            elif page=="Upload Image":
                st.markdown("<div class='title'>📸 Upload Farm Photo</div>", unsafe_allow_html=True)
                uploaded = st.file_uploader("Choose an image (jpg/png)", type=["jpg","png","jpeg"])
                if uploaded:
                    st.session_state.uploaded_image = uploaded.getvalue()
                    image = Image.open(io.BytesIO(st.session_state.uploaded_image)).convert("RGB")
                    st.image(image, caption="Preview", use_column_width=True)
                if st.button("Analyze Image"):
                    if not st.session_state.uploaded_image:
                        st.error("Upload image first")
                    else:
                        image = Image.open(io.BytesIO(st.session_state.uploaded_image)).convert("RGB")
                        res = detect_image(image)
                        st.session_state.analyze_result = res
                        st.success(f"Detected: {res['crop']} (conf {res['confidence']})")
                if st.session_state.analyze_result:
                    res = st.session_state.analyze_result
                    st.markdown(f"**Crop:** {res['crop']}  |  **Health:** {res['health']}")
                    notes = st.text_area("Notes (optional)")
                    if st.button("Submit Observation"):
                        img_name = save_image_bytes(st.session_state.uploaded_image)
                        rec = {
                            "submission_id": uuid.uuid4().hex,
                            "farmer_id": user['id'],
                            "farmer_name": user['name'],
                            "state": user['state'],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image_name": img_name,
                            "detected_crop": res['crop'],
                            "detected_health": res['health'],
                            "notes": notes,
                            "verified": "Pending",
                            "verifier": "",
                            "verify_ts": ""
                        }
                        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
                        save_subs(df)
                        st.success("Submitted — pending verification")
                        st.balloons()
            
            elif page=="Crop Info":
                st.markdown("<div class='title'>🌱 Crop Suggestions & Modern Techniques</div>", unsafe_allow_html=True)
                # Weather & crop suggestion
                coords = STATE_COORDS.get(user['state'], STATE_COORDS["Other"])
                weather_data = get_weather_data(*coords)
                
                if weather_data and 'current_weather' in weather_data:
                    temp = weather_data['current_weather'].get('temperature', 'N/A')
                    prec = weather_data['current_weather'].get('precipitation', 0)  # safe default
                    st.markdown(f"**Current Temperature:** {temp}°C | **Precipitation:** {prec}mm")
                    suggested_crop = suggest_crop(weather_data)
                    st.markdown(f"**Recommended Crop:** {suggested_crop}")
                    st.markdown("**Reason:** Based on region, current weather and forecast")
                else:
                    st.markdown("Weather data not available for your location.")
                
                st.markdown("**Modern Techniques:**")
                st.markdown("- Drip irrigation")
                st.markdown("- Organic fertilization")
                st.markdown("- Pest management")
                st.video("https://www.youtube.com/watch?v=FzC1LPXxYDM")
            
            elif page=="Notifications":
                st.markdown("<div class='title'>🔔 Notifications</div>", unsafe_allow_html=True)
                if not st.session_state.notification:
                    st.info("No notifications yet.")
                else:
                    for note in st.session_state.notification:
                        st.markdown(f"- {note}")
            
            elif page=="Government Schemes":
                st.markdown("<div class='title'>🏛️ Government Schemes</div>", unsafe_allow_html=True)
                schemes = [
                    "Pradhan Mantri Fasal Bima Yojana",
                    "Soil Health Card Scheme",
                    "PM-Kisan Samman Nidhi",
                    "National Agriculture Market (eNAM)"
                ]
                for s in schemes:
                    st.markdown(f"- {s}")

# ------------------- OFFICIAL -------------------
        else:
            if page=="Logout":
                st.session_state.user = None
                st.rerun()
            
            elif page=="Pending Verifications":
                st.markdown("<div class='title'>📋 Pending Verifications</div>", unsafe_allow_html=True)
                pending = df[df['verified']=="Pending"]
                if pending.empty:
                    st.info("No pending records.")
                else:
                    for idx,r in pending.iterrows():
                        st.markdown(f"**Farmer:** {r['farmer_name']} | Crop: {r['detected_crop']} | Health: {r['detected_health']}")
                        imgpath = os.path.join(DATA_DIR, r['image_name']) if r['image_name'] else None
                        if imgpath and os.path.exists(imgpath):
                            st.image(imgpath, width=250)
                        col1,col2 = st.columns([1,1])
                        with col1:
                            if st.button(f"Verify {r['submission_id']}", key=f"v{r['submission_id']}"):
                                df.loc[idx,'verified']="Verified"
                                df.loc[idx,'verifier']=user['name']
                                df.loc[idx,'verify_ts']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                save_subs(df)
                                st.success("Verified")
                                st.rerun()
                        with col2:
                            if st.button(f"Reject {r['submission_id']}", key=f"r{r['submission_id']}"):
                                df.loc[idx,'verified']="Rejected"
                                df.loc[idx,'verifier']=user['name']
                                df.loc[idx,'verify_ts']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                save_subs(df)
                                st.error("Rejected")
                                st.rerun()
            
            elif page=="Verified Records":
                st.markdown("<div class='title'>✅ Verified Records</div>", unsafe_allow_html=True)
                verified = df[df['verified']=="Verified"]
                if verified.empty:
                    st.info("No verified records yet.")
                else:
                    st.dataframe(verified)
