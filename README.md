# ICIRS — India-specific Cognitive Impairment Risk Score

Streamlit web app for community-level cognitive impairment screening.
Developed from LASI-DAD Wave 1–Wave 2 | IIPS Mumbai | 2024–25

## How to deploy on Streamlit Cloud

1. Create a GitHub repository
2. Upload `icirs_app.py` and `requirements.txt`
3. Go to https://share.streamlit.io
4. Connect your GitHub repo
5. Set main file path: `icirs_app.py`
6. Click Deploy

## How to run locally

```bash
pip install streamlit
streamlit run icirs_app.py
```

## ICIRS Performance
- AUC = 0.823 (optimism-corrected: 0.812)
- Youden cut-point: ≥13 points
- Sensitivity = 85.1% | Specificity = 68.2% | NPV = 98.2%
- Max score: 28 points | 9 items | No lab tests required
