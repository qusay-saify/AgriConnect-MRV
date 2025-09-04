# AgriConnect-MRV
AI-powered web app for farmers and officials to detect crop type &amp; health, provide weather-based crop suggestions, share modern techniques and government schemes, and enable MRV verification workflow, empowering smallholder farmers with actionable insights.

## Features

- Crop Detection & Health Monitoring (upload farm images)
- Real-time Weather Integration (state-specific)
- Crop Suggestions (based on weather, regional data, and profitability)
- Notifications & Government Schemes (alerts for farmers)
- Modern Farming Techniques (video tutorials)
- Farmer & Official Workflow (verification, MRV reporting)
- User-friendly UI/UX (high contrast text, intuitive navigation)


## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/qusay-saify/AgriConnect-MRV.git
   cd AgriConnect-MRV

2 Install dependencies

pip install -r requirements.txt

3 Run the app

streamlit run app.py




---

### **5️⃣ Optional: API Keys or Configuration**
If your weather API or other services require a key:

```markdown
## Configuration

- Open `app.py` and set your API key for Open-Meteo:
  ```python
  weather_api_key = "YOUR_API_KEY"



  
---

### **6️⃣ Organize Directory**
Explain folder structure so users understand where files are:

```markdown
## Directory Structure

AgriConnect-MRV/
├─ app.py               # Main Streamlit app
├─ requirements.txt     # Dependencies
├─ models/              # AI models for crop detection
├─ data/                # Uploaded images & CSV storage
└─ README.md

