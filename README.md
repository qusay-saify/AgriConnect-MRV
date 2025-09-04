# AgriConnect-MRV
AI-powered web app for farmers and officials to detect crop type &amp; health, provide weather-based crop suggestions, share modern techniques and government schemes, and enable MRV verification workflow, empowering smallholder farmers with actionable insights.

# AgriConnect-MRV 🌾

AI-powered web app for farmers and officials to monitor, verify, and optimize agriculture.

## Features
- Crop detection & health monitoring
- Weather-based crop suggestions
- Notifications & government schemes
- Farmer & official workflow
- Modern farming techniques
- User-friendly UI/UX



## Installation
1. Clone the repository:
```bash
git clone https://github.com/qusay-saify/AgriConnect-MRV.git
cd AgriConnect-MRV

2 Install dependencies:

pip install -r requirements.txt

3 Run the app:

streamlit run app.py


Directory Structure

AgriConnect-MRV/
├─ app.py
├─ requirements.txt
├─ models/
├─ data/
└─ README.md


workflow for AgriConnect-MRV

## System Workflow Flowchart

Farmer Login
     │
     ▼
Upload Farm Image & Region Info
     │
     ▼
AI Model Analysis
(Crop Detection & Disease Recognition)
     │
     ▼
Weather & Market Data Integration
     │
     ▼
Crop Suggestions & Recommendations
     │
     ▼
Notifications Module
(Govt Schemes & Modern Techniques)
     │
     ▼
Official Verification
(Approve / Reject Farmer Submission)
     │
     ▼
MRV Data Stored for Reporting


System Architecture

The AgriConnect-MRV platform is designed to be a farmer-friendly, scalable web application integrating AI, real-time data, and official verification workflows.

1. User Layer

Farmers: Upload farm images, receive crop health analysis, crop suggestions, notifications, and government scheme updates.

Officials: Verify farmer submissions, monitor data for MRV reporting and carbon credit assessments.

2. Application Layer (Streamlit Web App)

Frontend: Streamlit UI with high-contrast, intuitive interface, input forms, navigation buttons, and real-time updates.

Backend Logic:

AI models process farm images for crop detection and disease recognition.

Weather and market APIs provide real-time data for crop suggestions.

Notifications module sends alerts for government schemes and modern techniques.

Verification workflow for officials to approve/reject farmer submissions.

3. Data Layer

Local Storage / CSV: Stores farmer submissions, crop data, and verification results.

AI Models: Pretrained PyTorch models for crop detection and disease recognition.

External APIs:

Weather API (Open-Meteo) for real-time climate data.

Optional crop market API for profitability suggestions.

4. Workflow Overview

Farmer logs in → uploads farm image → selects region and crop type.

AI analyzes image → provides crop type, health status, and disease detection.

Weather & regional data → system suggests optimal crop to plant.

Notifications module → sends updates on government schemes & modern techniques.

Official logs in → verifies submissions → MRV data is recorded for reporting.

5. Scalability & Extensibility

Can integrate more AI models for multiple crop types and diseases.

Additional APIs for market trends, soil data, and satellite imagery.

Multi-language support for wider farmer adoption.