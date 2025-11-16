# SocialPulse: Social Media & E-commerce Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Issues](https://img.shields.io/github/issues/CodesavvySiddharth/SocialPulse)](https://github.com/CodesavvySiddharth/SocialPulse/issues)

## Overview
**SocialPulse** is a sentiment analysis tool that helps businesses monitor and understand public perception of their products and brands across multiple online platforms. Initially using datasets from **Amazon** and **Flipkart**, SocialPulse analyzes user reviews and comments to provide actionable insights.  

The system is designed to be **scalable**, allowing integration of additional e-commerce and social media datasets in the future for broader analysis.

---

## Key Features
- **Multi-Platform Sentiment Analysis**: Analyzes reviews and comments from various platforms to assess product or brand sentiment across different demographics.  
- **Real-Time Tracking & Alerts**: Monitors new reviews or mentions and highlights significant sentiment changes.  
- **Comprehensive Dashboard**: Displays sentiment trends, keyword analysis, and other metrics for strategic decision-making.  
- **Customizable Reports**: Generates detailed reports for specific timeframes, platforms, or sentiment categories.  
- **Future-Ready Dataset Integration**: Easily add more datasets from other e-commerce sites or social media platforms as needed.  

---

## Datasets
- **Current Datasets**:  
  - Amazon Reviews Dataset  
  - Flipkart Reviews Dataset  

- **Future Datasets**:  
  - Designed to integrate additional e-commerce platforms, social media platforms, or any other text-based review sources.  

---

## Built With
- **Python** – Backend development  
- **Streamlit** – Interactive web application  
- **VADER Sentiment** – Text sentiment scoring  
- **Altair / Matplotlib** – Visualizations  
- **Google Translate (googletrans)** – Lightweight multilingual support  
- **Google Gemini** – Final executive report generation  

---

### Getting Started

### Prerequisites
- Python 3.8 or higher  
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CodesavvySiddharth/SocialPulse.git
   cd SocialPulse
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure Gemini API key (choose one):
   - Option A (recommended): Streamlit secrets file
     ```bash
     cp .streamlit/secrets.example.toml .streamlit/secrets.toml
     # open .streamlit/secrets.toml and paste your key
     ```
     `.streamlit/secrets.toml` content:
     ```toml
     GEMINI_API_KEY = "your_api_key_here"
     ```
   - Option B: Environment variable
     ```bash
     export GEMINI_API_KEY="your_api_key_here"
     ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```
5. Open in your browser at:
   ```bash
   http://localhost:8501
   ```

---

## Usage
- Upload a CSV file in the “Upload a Review CSV File” section.
- Explore charts: Sentiment Distribution, Intent & Emotion Summaries, Aspect Breakdown.
- Review the “Aspect-Based Sentiment Summary (%)” table.
- Under “AI Analysis & Recommendations”, click “Final report generation” to trigger the Gemini-powered executive report.  
  - The report is displayed on-screen and can be downloaded as a `.txt` file.

### CSV Requirements
- Required columns:
  - `Review_Summary` (string): the review text
  - `Rating` (numeric): the associated rating (e.g., 1–5)
- Optional/ignored columns are allowed; any `Unnamed: 0` column will be dropped automatically.

### Notes
- You can toggle “Translate non-English reviews to English” in the sidebar.
- Category profiles (e.g., Electronics, Fashion) define aspect keywords and display names.

---

## Troubleshooting
- “Failed to generate report: GEMINI_API_KEY not found”: ensure the key is set via Streamlit secrets or environment variable as shown above.
- If translation is slow on large files, try disabling translation in the sidebar to speed up processing.

---

## Collaboration (Team Keys)
- Do not commit secrets. The repo includes:
  - `.streamlit/secrets.example.toml` (template)
  - `.gitignore` ignores `.streamlit/secrets.toml` and `.venv/`
- Each teammate should:
  1) Copy the template:
     ```bash
     cp .streamlit/secrets.example.toml .streamlit/secrets.toml
     ```
  2) Paste their own `GEMINI_API_KEY` into `.streamlit/secrets.toml`
  3) Run `streamlit run app.py`

### What to Commit
- Commit: `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `.streamlit/secrets.example.toml`
- Do not commit: `.streamlit/secrets.toml`, `.venv/`, `__pycache__/`, large datasets (keep samples only)
