"""
ICIRS — India-specific Cognitive Impairment Risk Score
Streamlit Web Application
Developed from: LASI-DAD Wave 1–Wave 2 | IIPS Mumbai | Ramnaresh (2024–25)
AUC = 0.823 | Optimism-corrected AUC = 0.812 | Youden cut-point = 13 pts
"""

import streamlit as st
import math

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICIRS — Cognitive Impairment Risk Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Main background */
  .main .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1000px; }

  /* Header banner */
  .header-banner {
    background: linear-gradient(135deg, #1F3864 0%, #2B6CB0 60%, #3182ce 100%);
    border-radius: 14px; padding: 2rem 2.5rem; margin-bottom: 1.8rem;
    color: white;
  }
  .header-banner h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem 0; color: white; }
  .header-banner p  { font-size: 0.92rem; margin: 0; opacity: 0.88; color: white; }

  /* Section cards */
  .section-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .section-title {
    font-size: 0.82rem; font-weight: 600; color: #2B6CB0;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 1rem; border-bottom: 2px solid #E6F0FA; padding-bottom: 0.5rem;
  }

  /* Score display */
  .score-box {
    border-radius: 14px; padding: 1.8rem; text-align: center;
    margin: 0.5rem 0 1rem 0;
  }
  .score-number { font-size: 4rem; font-weight: 700; line-height: 1; }
  .score-label  { font-size: 0.85rem; margin-top: 0.3rem; opacity: 0.8; }

  .risk-low      { background: linear-gradient(135deg,#E6FFED,#C6F6D5); color: #1E8449; border: 2px solid #48BB78; }
  .risk-moderate { background: linear-gradient(135deg,#FFFBEB,#FEFCBF); color: #975A16; border: 2px solid #F6AD55; }
  .risk-high     { background: linear-gradient(135deg,#FFF5F5,#FED7D7); color: #C53030; border: 2px solid #FC8181; }
  .risk-veryhigh { background: linear-gradient(135deg,#FFF5F5,#FEB2B2); color: #822727; border: 2px solid #E53E3E; }

  /* Risk label badge */
  .risk-badge {
    display: inline-block; border-radius: 20px; padding: 0.35rem 1.1rem;
    font-size: 1rem; font-weight: 600; margin-top: 0.6rem;
  }

  /* Progress bar container */
  .prog-container { background: #EDF2F7; border-radius: 10px; height: 14px;
    margin: 0.8rem 0; overflow: hidden; }
  .prog-fill { height: 100%; border-radius: 10px;
    transition: width 0.5s ease; }

  /* Item score row */
  .item-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.45rem 0; border-bottom: 1px solid #F0F4F8; font-size: 0.88rem;
  }
  .item-row:last-child { border-bottom: none; }
  .item-pts { font-weight: 600; }
  .pts-zero  { color: #A0AEC0; }
  .pts-low   { color: #48BB78; }
  .pts-mid   { color: #F6AD55; }
  .pts-high  { color: #E53E3E; }

  /* Recommendation box */
  .rec-box {
    border-radius: 10px; padding: 1rem 1.2rem; margin-top: 0.8rem;
    font-size: 0.9rem; line-height: 1.6;
  }
  .rec-low      { background: #F0FFF4; border-left: 4px solid #48BB78; }
  .rec-moderate { background: #FFFAF0; border-left: 4px solid #F6AD55; }
  .rec-high     { background: #FFF5F5; border-left: 4px solid #FC8181; }
  .rec-veryhigh { background: #FFF5F5; border-left: 4px solid #E53E3E; }

  /* Metric chips */
  .metric-chip {
    display: inline-block; background: #EBF4FF; color: #1A365D;
    border-radius: 8px; padding: 0.25rem 0.6rem; font-size: 0.78rem;
    font-weight: 500; margin: 0.2rem 0.15rem;
  }

  /* Info note */
  .info-note { background: #EBF8FF; border-left: 4px solid #63B3ED;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    font-size: 0.82rem; color: #2C5282; margin-top: 0.6rem; }

  /* Footer */
  .footer { text-align: center; font-size: 0.75rem; color: #A0AEC0;
    margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #EDF2F7; }

  /* Hide streamlit default elements */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  .stDeployButton {display: none;}

  /* Radio and selectbox styling */
  .stRadio > label { font-size: 0.88rem !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# ── ICIRS Scoring Logic ────────────────────────────────────────────────────────
def compute_icirs(age, illiterate, educ_yrs, iadl, infmem, midarm_cm, reads):
    """
    ICIRS scoring function — derived from LASI-DAD longitudinal analysis
    Returns: (total_score, item_breakdown_dict)

    Point weights (Framingham ratio method from logistic regression β):
      age_cat      : β=0.463  → 2 pts per 5-yr band above 60-64
      illiterate   : β=1.002  → 4 pts if cannot read
      educ_cat     : β=0.286  → 1 pt per education level (0-3)
      iadl_cat     : β=0.429  → 2 pts per IADL severity level (0-3)
      infmem       : β=0.723  → 3 pts if informant reports memory concern
      midarm_cat   : β=0.369  → 1 pt per nutritional risk level (0-3)
      reading_score: β=0.274  → 1 pt if does NOT read regularly
      adl, married : β<0 (protective) → 0 pts
    Maximum score = 28 points. Youden cut-point = 13.
    """

    # Age category (0–4)
    if age < 65:
        age_cat = 0
    elif age < 70:
        age_cat = 1
    elif age < 75:
        age_cat = 2
    elif age < 80:
        age_cat = 3
    else:
        age_cat = 4
    age_pts = age_cat * 2

    # Illiteracy (0 or 4)
    illit_pts = 4 if illiterate else 0

    # Education category (0–3) → 1 pt per level
    if educ_yrs > 8:
        educ_cat = 0
    elif educ_yrs >= 5:
        educ_cat = 1
    elif educ_yrs >= 1:
        educ_cat = 2
    else:
        educ_cat = 3
    educ_pts = educ_cat * 1

    # IADL category (0–3) → 2 pts per level
    if iadl == 0:
        iadl_cat = 0
    elif iadl <= 2:
        iadl_cat = 1
    elif iadl <= 4:
        iadl_cat = 2
    else:
        iadl_cat = 3
    iadl_pts = iadl_cat * 2

    # Informant memory (0 or 3)
    infmem_pts = 3 if infmem else 0

    # Mid-arm circumference (0–3) → 1 pt per nutritional risk level
    if midarm_cm > 26:
        midarm_cat = 0
    elif midarm_cm >= 24:
        midarm_cat = 1
    elif midarm_cm >= 22:
        midarm_cat = 2
    else:
        midarm_cat = 3
    midarm_pts = midarm_cat * 1

    # Not reading (0 or 1)
    reading_pts = 0 if reads else 1

    total = age_pts + illit_pts + educ_pts + iadl_pts + infmem_pts + midarm_pts + reading_pts

    breakdown = {
        "Age category":               {"pts": age_pts,    "max": 8,  "detail": f"Age {age} → band {age_cat}"},
        "Illiteracy":                  {"pts": illit_pts,  "max": 4,  "detail": "Cannot read" if illiterate else "Can read"},
        "Education level":             {"pts": educ_pts,   "max": 3,  "detail": f"{educ_yrs} yrs → level {educ_cat}"},
        "IADL limitations":            {"pts": iadl_pts,   "max": 6,  "detail": f"{iadl} limitations → level {iadl_cat}"},
        "Informant memory concern":    {"pts": infmem_pts, "max": 3,  "detail": "Concern reported" if infmem else "No concern"},
        "Mid-arm circumference":       {"pts": midarm_pts, "max": 3,  "detail": f"{midarm_cm} cm → level {midarm_cat}"},
        "Not reading regularly":       {"pts": reading_pts,"max": 1,  "detail": "Does not read" if not reads else "Reads regularly"},
    }

    return total, breakdown


def risk_category(score):
    """Returns (label, css_class, colour, recommendation, action)"""
    if score <= 6:
        return (
            "Low Risk",
            "risk-low",
            "#1E8449",
            "This individual's ICIRS score is below the high-risk threshold. "
            "Routine annual cognitive check-up is recommended. Encourage physical activity, "
            "social engagement, and reading habits.",
            [
                "✅ Annual cognitive health check-up",
                "📚 Encourage reading and mental stimulation",
                "🏃 Promote regular physical activity and outdoor exposure",
                "🥗 Ensure adequate nutrition (mid-arm circumference monitoring)",
                "📅 Reassess in 12 months",
            ]
        )
    elif score <= 12:
        return (
            "Moderate Risk",
            "risk-moderate",
            "#975A16",
            "This individual's score is approaching the Youden cut-point of 13. "
            "Increased vigilance and a brief structured memory assessment is recommended "
            "within the next 6 months.",
            [
                "⚠️ Brief structured memory test (e.g., Mini-Mental State or MoCA) within 6 months",
                "📋 Document informant concerns if any arise",
                "🥗 Nutritional assessment if mid-arm circumference < 24 cm",
                "🤝 Social support evaluation (loneliness screening)",
                "📅 Reassess ICIRS in 6 months",
            ]
        )
    elif score <= 18:
        return (
            "High Risk",
            "risk-high",
            "#C53030",
            "This individual scores at or above the Youden cut-point (≥13). "
            "The ICIRS model has 85.1% sensitivity at this cut-point — 8 in 10 individuals "
            "who will develop cognitive impairment within 3–4 years score here. "
            "Prompt referral for formal cognitive assessment is strongly recommended.",
            [
                "🔴 Refer for formal neuropsychological evaluation or memory clinic",
                "📋 Administer brief cognitive screening (MoCA / Mini-Cog) immediately",
                "👨‍⚕️ Medical review of reversible causes (thyroid, B12, depression, medications)",
                "🥗 Nutritional support if mid-arm circumference < 24 cm",
                "🏥 Inform primary care physician and document in health record",
                "👨‍👩‍👧 Family counselling and caregiver support",
                "📅 Re-evaluate in 3 months",
            ]
        )
    else:
        return (
            "Very High Risk",
            "risk-veryhigh",
            "#822727",
            "This individual's score is in the very high range (≥19 points). "
            "This represents the extreme end of the ICIRS distribution, overlapping with "
            "the Extreme-risk cluster (CI rate=14.6%, adjusted OR=6.61 vs lowest-risk cluster). "
            "Urgent specialist referral is warranted.",
            [
                "🚨 Urgent referral to neurologist or geriatric psychiatrist",
                "📋 Immediate comprehensive cognitive assessment (full neuropsychology battery)",
                "🧬 Consider brain imaging and blood workup (thyroid, B12, homocysteine)",
                "👨‍👩‍👧 Immediate caregiver involvement and safety planning",
                "📑 Review medications for cognition-impairing drugs",
                "🏥 Consider day care / community mental health team involvement",
                "📅 Follow-up within 4–6 weeks",
            ]
        )


def pt_color_class(pts, max_pts):
    if pts == 0:
        return "pts-zero"
    ratio = pts / max_pts if max_pts > 0 else 0
    if ratio <= 0.25:
        return "pts-low"
    elif ratio <= 0.6:
        return "pts-mid"
    else:
        return "pts-high"


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>🧠 ICIRS — Cognitive Impairment Risk Screener</h1>
  <p>
    India-specific Cognitive Impairment Risk Score &nbsp;|&nbsp;
    Developed from LASI-DAD Wave 1–Wave 2 longitudinal data &nbsp;|&nbsp;
    AUC = 0.823 &nbsp;|&nbsp; Optimism-corrected AUC = 0.812 &nbsp;|&nbsp;
    IIPS Mumbai · Ramnaresh (2024–25)
  </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋  Risk Assessment", "ℹ️  About ICIRS", "📊  Score Reference"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    # ── LEFT: Input form ──────────────────────────────────────────────────────
    with col_form:

        # Section 1 — Demographics
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)

        age = st.number_input(
            "Age (years)",
            min_value=60, max_value=100, value=68, step=1,
            help="Enter age in completed years. ICIRS is validated for adults aged 60+."
        )

        sex = st.radio("Sex", ["Male", "Female"], horizontal=True,
                       help="Used for subgroup-adjusted interpretation only (not in scoring).")

        residence = st.radio("Residence", ["Urban", "Rural"], horizontal=True,
                             help="Used for subgroup-adjusted interpretation only.")

        caste = st.selectbox("Caste / social group",
                             ["General / OBC", "Scheduled Caste (SC)", "Scheduled Tribe (ST)"],
                             help="Used for subgroup-adjusted interpretation only.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Section 2 — Education & Literacy
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📚 Education & Literacy</div>', unsafe_allow_html=True)

        illiterate = st.radio(
            "Can this person read?",
            ["Yes — can read", "No — cannot read (illiterate)"],
            help="Illiteracy is the highest-weighted ICIRS item (4 points, β=1.002). "
                 "Crude OR for incident CI in illiterate vs literate elderly: 7.08 (LASI-DAD)."
        )
        illiterate_bool = illiterate == "No — cannot read (illiterate)"

        educ_yrs = st.slider(
            "Years of formal education completed",
            min_value=0, max_value=20, value=5, step=1,
            help="Include all formal schooling years. 0 = never attended school."
        )

        reads = st.radio(
            "Does this person read regularly? (books, newspapers, phone)",
            ["Yes — reads regularly", "No — does not read regularly"],
            help="Regular reading is a protective lifestyle activity. "
                 "Standardised β = +0.091 in the continuous cognitive change model."
        )
        reads_bool = reads == "Yes — reads regularly"
        st.markdown('</div>', unsafe_allow_html=True)

        # Section 3 — Functional Status
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🦽 Functional Status</div>', unsafe_allow_html=True)

        st.markdown("**Instrumental Activities of Daily Living (IADL)**")
        st.caption("Count difficulties with: preparing food, managing money, using transport, "
                   "shopping, managing medications, using telephone, housework, laundry.")

        iadl = st.slider(
            "Number of IADL limitations (out of 7)",
            min_value=0, max_value=7, value=1, step=1,
            help="IADL limitations: adjusted OR=1.239 per limitation in Objective 4 "
                 "logistic model (FDR-significant, p<0.001)."
        )

        st.markdown("**ADL limitations** *(informational — not scored)*")
        st.caption("Basic activities: bathing, dressing, eating, toileting, transferring, continence.")
        adl = st.slider(
            "Number of ADL limitations (out of 6) — not scored (protective β)",
            min_value=0, max_value=6, value=0, step=1,
            help="ADL limitations had a negative β coefficient (−0.404) in the logistic model, "
                 "suggesting a protective or survivorship effect. Assigned 0 points in ICIRS."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Section 4 — Nutrition
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🥗 Nutritional Status</div>', unsafe_allow_html=True)

        midarm_known = st.radio(
            "Is mid-arm circumference measurement available?",
            ["Yes — I have the measurement", "No — not measured"],
            help="Mid-arm circumference is a simple tape-measure nutritional indicator. "
                 "Correlates with protein-energy malnutrition. Effect: std-β=+0.105 in "
                 "continuous cognitive change model."
        )

        if midarm_known == "Yes — I have the measurement":
            midarm_cm = st.slider(
                "Mid-arm circumference (cm)",
                min_value=15.0, max_value=40.0, value=26.0, step=0.5,
                help=">26 cm: adequate nutrition (0 pts). 24–26 cm: mild risk (1 pt). "
                     "22–24 cm: moderate risk (2 pts). <22 cm: severe malnutrition risk (3 pts)."
            )
        else:
            midarm_cm = 26.1  # assign 0 pts (assume adequate) with note
            st.info("Mid-arm circumference not available — assuming >26 cm (0 points). "
                    "Measure with a tape measure for accurate scoring.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Section 5 — Informant Report
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👨‍👩‍👧 Informant Memory Assessment</div>', unsafe_allow_html=True)

        st.markdown(
            "Ask a family member, caregiver, or close companion who knows the person well:"
        )
        st.markdown(
            "> *\"Have you noticed any memory problems or forgetfulness in this person "
            "over the past year? Does their memory seem worse than others their age?\"*"
        )

        infmem = st.radio(
            "Informant's response:",
            ["No — no memory concerns reported",
             "Yes — informant reports memory concern"],
            help="Informant memory concern: adjusted OR=2.142 (95% CI 1.199–3.826) in "
                 "Objective 4 full model (FDR-significant, p=0.010). This is the "
                 "strongest single non-cognitive predictor in the LASI-DAD dataset. "
                 "All members of the Extreme-risk GMM cluster (CI rate=14.6%) had this flag."
        )
        infmem_bool = infmem == "Yes — informant reports memory concern"
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: Results panel ──────────────────────────────────────────────────
    with col_result:
        # Compute score
        total_score, breakdown = compute_icirs(
            age=age,
            illiterate=illiterate_bool,
            educ_yrs=educ_yrs,
            iadl=iadl,
            infmem=infmem_bool,
            midarm_cm=midarm_cm,
            reads=reads_bool,
        )

        label, css_class, colour, rec_text, actions = risk_category(total_score)
        rec_css = css_class.replace("risk-", "rec-")

        # ── Score display ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="score-box {css_class}">
          <div class="score-number">{total_score}</div>
          <div class="score-label">out of 28 points</div>
          <div class="risk-badge" style="background:{colour}22; color:{colour}; border:1.5px solid {colour}aa;">
            {label}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar
        pct = total_score / 28 * 100
        bar_colors = {"risk-low":"#48BB78","risk-moderate":"#F6AD55",
                      "risk-high":"#FC8181","risk-veryhigh":"#E53E3E"}
        bar_col = bar_colors.get(css_class, "#4299E1")
        st.markdown(f"""
        <div class="prog-container">
          <div class="prog-fill" style="width:{pct:.1f}%; background:{bar_col};"></div>
        </div>
        <div style="font-size:0.75rem;color:#718096;display:flex;justify-content:space-between;">
          <span>0 pts</span><span>Cut-point: 13</span><span>28 pts</span>
        </div>
        """, unsafe_allow_html=True)

        # Cutpoint indicator
        if total_score >= 13:
            st.markdown(f"""
            <div style="margin-top:0.6rem;padding:0.5rem 0.8rem;background:#FFF5F5;
                border:1px solid #FC8181;border-radius:8px;font-size:0.83rem;color:#C53030;">
              ⚠️ Score ≥13 (Youden cut-point): <strong>Positive screen</strong><br>
              Sensitivity = 85.1% &nbsp;|&nbsp; Specificity = 68.2% &nbsp;|&nbsp; NPV = 98.2%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="margin-top:0.6rem;padding:0.5rem 0.8rem;background:#F0FFF4;
                border:1px solid #48BB78;border-radius:8px;font-size:0.83rem;color:#1E8449;">
              ✅ Score &lt;13 (below Youden cut-point): <strong>Negative screen</strong><br>
              NPV = 98.2% — 98.2% of individuals below this threshold will remain cognitively intact
            </div>
            """, unsafe_allow_html=True)

        # ── Score breakdown ────────────────────────────────────────────────────
        st.markdown('<div class="section-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Score Breakdown</div>', unsafe_allow_html=True)

        for item, vals in breakdown.items():
            pts   = vals["pts"]
            maxp  = vals["max"]
            detail = vals["detail"]
            col_cls = pt_color_class(pts, maxp)
            st.markdown(f"""
            <div class="item-row">
              <div>
                <span style="font-weight:500;">{item}</span><br>
                <span style="font-size:0.77rem;color:#718096;">{detail}</span>
              </div>
              <div class="item-pts {col_cls}">{pts}/{maxp} pts</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="item-row" style="border-top:2px solid #E2E8F0;margin-top:0.3rem;
             padding-top:0.6rem;font-weight:700;">
          <div>TOTAL ICIRS SCORE</div>
          <div style="font-size:1.1rem;color:{colour};">{total_score} / 28</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Subgroup-adjusted note ─────────────────────────────────────────────
        is_sc_st = caste in ["Scheduled Caste (SC)", "Scheduled Tribe (ST)"]
        is_illiterate = illiterate_bool
        is_female = sex == "Female"
        is_rural = residence == "Rural"

        subgroup_notes = []
        subgroup_auc = 0.823  # baseline
        adj_cutpoint = 13

        if is_female:
            subgroup_auc = min(subgroup_auc, 0.767)
            subgroup_notes.append("Female subgroup: AUC=0.767 (lower than male 0.889). "
                                  "Consider cut-point ≥12 for improved specificity.")
            adj_cutpoint = 12
        if is_illiterate:
            subgroup_auc = min(subgroup_auc, 0.736)
            subgroup_notes.append("Illiterate subgroup: AUC=0.736. "
                                  "Consider cut-point ≥11 for improved specificity (current spec=41.9%).")
            adj_cutpoint = min(adj_cutpoint, 11)
        if is_sc_st:
            subgroup_auc = min(subgroup_auc, 0.765)
            subgroup_notes.append("SC/ST subgroup: AUC=0.765. "
                                  "Consider cut-point ≥12 for equitable performance.")
            adj_cutpoint = min(adj_cutpoint, 12)
        if is_rural:
            subgroup_notes.append("Rural subgroup: AUC=0.802. Overall performance maintained.")

        if subgroup_notes:
            st.markdown('<div class="info-note">', unsafe_allow_html=True)
            st.markdown(f"**📌 Subgroup-adjusted note** (AUC for this profile: {subgroup_auc:.3f}):")
            for note in subgroup_notes:
                st.markdown(f"• {note}")
            if adj_cutpoint < 13:
                flag = total_score >= adj_cutpoint
                st.markdown(f"**Adjusted cut-point for this individual: ≥{adj_cutpoint} pts** → "
                             f"{'⚠️ Positive screen' if flag else '✅ Negative screen'}")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Clinical recommendation ────────────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💊 Clinical Recommendation</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box {rec_css}">{rec_text}</div>',
                    unsafe_allow_html=True)
        st.markdown("**Action checklist:**")
        for action in actions:
            st.markdown(f"- {action}")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown("""
        <div class="info-note" style="margin-top:0.8rem;">
          <strong>⚕️ Clinical Disclaimer:</strong> ICIRS is a <em>screening tool</em>, not a
          diagnostic instrument. A positive screen does not confirm cognitive impairment.
          All high-risk individuals should be referred for formal clinical assessment.
          This tool should not replace clinical judgement.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📖 What is ICIRS?</div>', unsafe_allow_html=True)
        st.markdown("""
ICIRS (India-specific Cognitive Impairment Risk Score) is the first
community-deployable cognitive screening tool derived from Indian longitudinal data.

It was developed from the **LASI-DAD** (Longitudinal Ageing Study in India —
Diagnostic Assessment of Dementia) **Wave 1–Wave 2** cohort:

- **N = 1,869** cognitively intact elderly (aged 60+) at Wave 1 (2017–18)
- **148 incident CI cases** (7.9%) identified at Wave 2 (2020–21)
- Follow-up period: approximately 3.5 years
- Nationally representative stratified multi-stage cluster sampling

**Why India-specific?** Western scores (CAIDE, ANU-ADRI) were developed in
Finnish and Australian populations where illiteracy is rare, malnutrition uncommon,
and caste-based social stratification absent. When applied to LASI-DAD:
- CAIDE AUC = 0.718 (sensitivity = 62.2%)
- ANU-ADRI AUC = 0.740 (sensitivity = 65.5%)
- **ICIRS AUC = 0.823 (sensitivity = 85.1%)**
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔬 Methodology</div>', unsafe_allow_html=True)
        st.markdown("""
**Score Development:**
- Predictors selected from Objective 4 Benjamini-Hochberg FDR-corrected results
  (16 tests surviving FDR across 33 logistic + 33 OLS = 66 simultaneous tests)
- Community-measurable items only (no blood tests required)
- Point weights derived using Framingham ratio method
  (β / β_min, rounded to nearest integer)

**Validation:**
- 500-resample bootstrap optimism correction (Harrell et al., 1996)
- Hosmer-Lemeshow calibration: χ²=6.44, p=0.598 (good)
- Brier score = 0.063

**Cluster Analysis:**
- Gaussian Mixture Model (GMM) on 10 FDR-robust predictors
- BIC-optimal k=6 clusters; CI rates ranged 2.1%–14.6%
- Spearman rank agreement with logistic ΔR²: ρ=0.821 (p=0.023)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Performance Summary</div>',
                    unsafe_allow_html=True)

        metrics = {
            "Naive AUC":               "0.823",
            "Optimism-corrected AUC":  "0.812",
            "Sensitivity (cut=13)":    "85.1%",
            "Specificity (cut=13)":    "68.2%",
            "PPV (prevalence=7.9%)":   "18.7%",
            "NPV":                     "98.2%",
            "HL p-value (calibration)":"0.598",
            "Brier score":             "0.063",
            "Bootstrap resamples":     "500",
            "Cut-point (Youden)":      "≥ 13 points",
        }
        for k, v in metrics.items():
            st.markdown(f"""
            <div class="item-row">
              <span>{k}</span>
              <strong style="color:#2B6CB0;">{v}</strong>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👥 Subgroup AUC Performance</div>',
                    unsafe_allow_html=True)

        subgroups = [
            ("Overall",       "All",         1869, "0.823"),
            ("Gender",        "Male",         848, "0.889 ✅"),
            ("Gender",        "Female",      1021, "0.767 ⚠️"),
            ("Residence",     "Urban",        711, "0.838"),
            ("Residence",     "Rural",       1158, "0.802"),
            ("Education",     "Literate",     870, "0.812"),
            ("Education",     "Illiterate",   999, "0.736 ⚠️"),
            ("Caste",         "General/OBC", 1459, "0.841"),
            ("Caste",         "SC/ST",        410, "0.765 ⚠️"),
        ]
        for strat, sub, n, auc_s in subgroups:
            st.markdown(f"""
            <div class="item-row">
              <span><em style="color:#718096;">{strat}</em> — {sub} (n={n:,})</span>
              <strong style="color:#2B6CB0;">{auc_s}</strong>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-note" style="margin-top:0.8rem;">
          ⚠️ = AUC meaningfully below overall (Δ&gt;0.05). Subgroup-specific
          cut-points (≥11 illiterate; ≥12 SC/ST and female) improve equity.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏥 Intended Use</div>', unsafe_allow_html=True)
        st.markdown("""
ICIRS is designed for use by:
- **ASHA workers** and community health volunteers
- **Primary care nurses** and ANMs
- **Geriatricians** for rapid pre-screening
- **Researchers** for population-level CI risk estimation

Administration time: **~5–7 minutes**
Equipment needed: tape measure (for mid-arm circumference)

**Not intended for:**
- Clinical diagnosis of dementia or MCI
- Replacement of neuropsychological assessment
- Monitoring of treatment response
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SCORE REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_r1, col_r2 = st.columns(2, gap="large")

    with col_r1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 ICIRS Scoring Card</div>',
                    unsafe_allow_html=True)
        st.markdown("""
*Print or laminate this card for field use.*

| Item | Categories | Points |
|------|-----------|--------|
| **Age** | 60–64 yrs | 0 |
| | 65–69 yrs | 2 |
| | 70–74 yrs | 4 |
| | 75–79 yrs | 6 |
| | ≥80 yrs | 8 |
| **Illiteracy** | Can read | 0 |
| | Cannot read | **4** |
| **Education** | >8 years | 0 |
| | 5–8 years | 1 |
| | 1–4 years | 2 |
| | No education | 3 |
| **IADL** | No limitations | 0 |
| | 1–2 limitations | 2 |
| | 3–4 limitations | 4 |
| | ≥5 limitations | 6 |
| **Informant memory** | No concern | 0 |
| | Concern reported | **3** |
| **Mid-arm circ.** | >26 cm | 0 |
| | 24–26 cm | 1 |
| | 22–24 cm | 2 |
| | <22 cm | 3 |
| **Not reading** | Reads regularly | 0 |
| | Does not read | 1 |
| | | |
| **TOTAL** | | **/ 28** |
        """)
        st.markdown("""
---
**Interpretation:**
- Score **0–6**: 🟢 Low Risk
- Score **7–12**: 🟡 Moderate Risk
- Score **≥13**: 🔴 **High Risk** — Refer for assessment
- Score **≥19**: 🔴 **Very High Risk** — Urgent referral
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 IADL Assessment Guide</div>',
                    unsafe_allow_html=True)
        st.markdown("""
Ask about difficulty or inability to perform **each of the following**:

1. **Preparing food** — Can they cook a meal alone?
2. **Managing money** — Can they handle cash, pay bills?
3. **Using transport** — Bus/auto alone without help?
4. **Shopping** — Buy groceries independently?
5. **Managing medications** — Take correct pills at correct time?
6. **Using telephone** — Dial numbers, use mobile?
7. **Housework** — Sweep, mop, basic cleaning?

Count the number of items where the person has **difficulty or needs help**.
Score 0 = fully independent; 7 = dependent in all areas.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📏 Mid-Arm Circumference Guide</div>',
                    unsafe_allow_html=True)
        st.markdown("""
**How to measure:**
1. Ask the person to relax their arm at their side
2. Find the midpoint between shoulder tip and elbow tip
3. Wrap tape measure horizontally around the arm at the midpoint
4. Do not compress the skin — tape should be snug but not tight
5. Read to nearest 0.5 cm

**Interpretation:**
| MAC | Nutritional status | ICIRS points |
|-----|-------------------|-------------|
| >26 cm | Adequate | 0 |
| 24–26 cm | Mild risk | 1 |
| 22–24 cm | Moderate risk | 2 |
| <22 cm | Severe malnutrition risk | 3 |

*Normal reference: Men >23 cm; Women >22 cm (WHO)*
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧬 GMM Risk Cluster Reference</div>',
                    unsafe_allow_html=True)
        st.markdown("""
Six latent risk profiles from LASI-DAD (k=6 GMM):

| Cluster | Name | n | CI Rate | Key feature |
|---------|------|---|---------|-------------|
| 1 | Lower-risk | 373 | **2.1%** | High education, all read |
| 2 | Moderate-risk | 251 | 2.8% | High education, all read |
| 3 | Intermediate-risk | 389 | 8.7% | Illiterate, married, 0% read |
| 4 | Higher-risk | 337 | 9.2% | Illiterate, married, 0% read |
| 5 | Very high-risk | 361 | 12.5% | Illiterate, 0% married, 87% female |
| 6 | Extreme-risk | 158 | **14.6%** | 100% informant memory concern |

*Cluster 2 vs Cluster 1: OR=1.05 (p=0.925, ns)*
*Cluster 6 vs Cluster 1: OR=6.61 (p<0.001)***
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <strong>ICIRS</strong> — India-specific Cognitive Impairment Risk Score &nbsp;|&nbsp;
  Developed by Ramnaresh, M.Sc. Survey Research & Data Analytics,
  International Institute for Population Sciences, Mumbai, 2024–25 &nbsp;|&nbsp;
  Based on LASI-DAD Wave 1–Wave 2 data (N=1,869; incident CI=148; AUC=0.823)<br>
  <em>For research and screening purposes only. Not a diagnostic tool.
  Always refer high-risk individuals for formal clinical assessment.</em>
</div>
""", unsafe_allow_html=True)
