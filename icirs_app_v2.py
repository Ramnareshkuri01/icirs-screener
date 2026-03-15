"""
ICIRS-Extended — India-specific Cognitive Impairment Risk Score
Streamlit Web Application  |  Version 2.0  |  Extended 14-variable model

Model: Logistic regression (class_weight=balanced, C=0.5)
Trained on: LASI-DAD Wave 1–Wave 2 | N=1,869 | Incident CI=148 (7.9%)
Extended AUC = 0.829 | vs original 7-var integer score AUC = 0.821
Subgroup-specific Youden thresholds for sex / rural / education / caste

Variables used:
  Core (from Obj4 FDR analysis): age, illiteracy, education, IADL, informant
    memory, mid-arm circumference, reading habits
  Extended (previously absent): sex, rural/urban, SC/ST caste,
    depression (CESD), BMI, exercise, marital status

IIPS Mumbai | Ramnaresh (2024-25)
"""

import streamlit as st
import math

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICIRS — Cognitive Impairment Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main .block-container{padding:1.5rem 2rem 3rem 2rem;max-width:1080px;}

.header-banner{background:linear-gradient(135deg,#1F3864 0%,#2B6CB0 60%,#3182ce 100%);
  border-radius:14px;padding:1.8rem 2.5rem;margin-bottom:1.6rem;color:white;}
.header-banner h1{font-size:1.85rem;font-weight:700;margin:0 0 0.3rem 0;color:white;}
.header-banner p{font-size:0.88rem;margin:0;opacity:0.9;color:white;}

.section-card{background:white;border:1px solid #e2e8f0;border-radius:12px;
  padding:1.3rem 1.5rem;margin-bottom:1.1rem;box-shadow:0 1px 4px rgba(0,0,0,0.05);}
.section-title{font-size:0.79rem;font-weight:700;color:#2B6CB0;text-transform:uppercase;
  letter-spacing:0.09em;margin-bottom:0.9rem;border-bottom:2px solid #E6F0FA;
  padding-bottom:0.45rem;}

.score-box{border-radius:14px;padding:1.6rem;text-align:center;margin:0.4rem 0 0.8rem 0;}
.score-number{font-size:3.6rem;font-weight:700;line-height:1;}
.score-label{font-size:0.82rem;margin-top:0.2rem;opacity:0.75;}
.risk-badge{display:inline-block;border-radius:20px;padding:0.3rem 1rem;
  font-size:0.95rem;font-weight:600;margin-top:0.5rem;}

.risk-low     {background:linear-gradient(135deg,#E6FFED,#C6F6D5);color:#1E8449;border:2px solid #48BB78;}
.risk-moderate{background:linear-gradient(135deg,#FFFBEB,#FEFCBF);color:#975A16;border:2px solid #F6AD55;}
.risk-high    {background:linear-gradient(135deg,#FFF5F5,#FED7D7);color:#C53030;border:2px solid #FC8181;}
.risk-veryhigh{background:linear-gradient(135deg,#FFF5F5,#FEB2B2);color:#822727;border:2px solid #E53E3E;}

.prog-container{background:#EDF2F7;border-radius:10px;height:12px;
  margin:0.7rem 0;overflow:hidden;}
.prog-fill{height:100%;border-radius:10px;}

.item-row{display:flex;justify-content:space-between;align-items:center;
  padding:0.4rem 0;border-bottom:1px solid #F0F4F8;font-size:0.86rem;}
.item-row:last-child{border-bottom:none;}
.item-pts{font-weight:600;}
.pts-zero{color:#A0AEC0;} .pts-low{color:#48BB78;}
.pts-mid{color:#F6AD55;} .pts-high{color:#E53E3E;}

.rec-box{border-radius:8px;padding:0.9rem 1.1rem;margin-top:0.7rem;
  font-size:0.88rem;line-height:1.65;}
.rec-low      {background:#F0FFF4;border-left:4px solid #48BB78;}
.rec-moderate {background:#FFFAF0;border-left:4px solid #F6AD55;}
.rec-high     {background:#FFF5F5;border-left:4px solid #FC8181;}
.rec-veryhigh {background:#FFF5F5;border-left:4px solid #E53E3E;}

.info-note{background:#EBF8FF;border-left:4px solid #63B3ED;border-radius:0 8px 8px 0;
  padding:0.75rem 1rem;font-size:0.81rem;color:#2C5282;margin-top:0.5rem;}
.warn-note{background:#FFFAF0;border-left:4px solid #F6AD55;border-radius:0 8px 8px 0;
  padding:0.75rem 1rem;font-size:0.81rem;color:#7B341E;margin-top:0.5rem;}
.success-note{background:#F0FFF4;border-left:4px solid #48BB78;border-radius:0 8px 8px 0;
  padding:0.75rem 1rem;font-size:0.81rem;color:#1E8449;margin-top:0.5rem;}

.prob-gauge{position:relative;height:28px;background:#EDF2F7;border-radius:14px;
  overflow:hidden;margin:0.5rem 0;}
.prob-fill{position:absolute;left:0;top:0;height:100%;border-radius:14px;
  display:flex;align-items:center;padding-left:10px;
  font-size:0.78rem;font-weight:600;color:white;}

.footer{text-align:center;font-size:0.73rem;color:#A0AEC0;margin-top:2rem;
  padding-top:0.8rem;border-top:1px solid #EDF2F7;}
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
.stDeployButton{display:none;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED MODEL — Logistic regression coefficients
# Trained on LASI-DAD N=1,869 | class_weight=balanced | C=0.5
# AUC=0.829 (vs 0.821 for 7-variable integer ICIRS)
# ═══════════════════════════════════════════════════════════════════════════════

MODEL = {
    "intercept":    -4.355357,
    # Core FDR-robust predictors
    "age":           0.090058,   # per year
    "illiterate":    0.836345,   # 1=cannot read
    "educ_yrs":     -0.132392,   # per year (protective)
    "iadl":          0.202763,   # per limitation (0-7)
    "infmem":        0.579470,   # 1=concern reported
    "midarm":       -0.104797,   # per cm (protective — lower = more risk)
    "reading":      -0.066144,   # 1=reads regularly (protective)
    # Extended — sex, residence, caste, mental health, lifestyle
    "female":        0.205478,   # 1=female (higher risk — mediated by educ gap)
    "rural":         0.152205,   # 1=rural
    "sc_st":         0.374961,   # 1=SC/ST caste
    "depression":   -0.024987,   # CESD-10 score (weak — near zero)
    "bmi":          -0.004765,   # per kg/m² (very weak)
    "exercise":      0.030585,   # 1=exercises (weak, non-significant)
    "married":      -0.155043,   # 1=married (protective)
}

# Subgroup-specific Youden-optimal probability thresholds
# From per-subgroup ROC analysis on LASI-DAD training data
THRESHOLDS = {
    "overall":    0.4963,
    "male":       0.4051,
    "female":     0.5054,
    "urban":      0.3010,
    "rural":      0.5251,
    "literate":   0.2678,
    "illiterate": 0.6207,
    "general":    0.4963,
    "sc_st":      0.5940,
}

# Subgroup AUC reference
SUBGROUP_AUC = {
    "Overall":     0.829,
    "Male":        0.895,
    "Female":      0.774,
    "Urban":       0.859,
    "Rural":       0.798,
    "Literate":    0.852,
    "Illiterate":  0.739,
    "General/OBC": 0.841,
    "SC/ST":       0.781,
}


def compute_probability(age, illiterate, educ_yrs, iadl, infmem, midarm_cm,
                        reads, female, rural, sc_st, depression, bmi,
                        exercise, married):
    """
    Compute predicted probability of incident CI using the extended
    logistic regression model trained on LASI-DAD Wave 1-Wave 2.
    Returns probability in [0, 1].
    """
    logit = (
        MODEL["intercept"]
        + MODEL["age"]       * age
        + MODEL["illiterate"]* (1 if illiterate else 0)
        + MODEL["educ_yrs"]  * educ_yrs
        + MODEL["iadl"]      * iadl
        + MODEL["infmem"]    * (1 if infmem else 0)
        + MODEL["midarm"]    * midarm_cm
        + MODEL["reading"]   * (1 if reads else 0)
        + MODEL["female"]    * (1 if female else 0)
        + MODEL["rural"]     * (1 if rural else 0)
        + MODEL["sc_st"]     * (1 if sc_st else 0)
        + MODEL["depression"]* depression
        + MODEL["bmi"]       * bmi
        + MODEL["exercise"]  * (1 if exercise else 0)
        + MODEL["married"]   * (1 if married else 0)
    )
    return 1 / (1 + math.exp(-logit))


def get_subgroup_key(female, rural, illiterate, sc_st):
    """
    Determine the primary subgroup key for threshold selection.
    Priority: illiterate > SC/ST > female > rural > overall
    (most vulnerable group's threshold takes priority)
    """
    if illiterate:
        return "illiterate"
    elif sc_st:
        return "sc_st"
    elif female and rural:
        return "rural"  # rural threshold is more conservative for rural women
    elif female:
        return "female"
    elif rural:
        return "rural"
    else:
        return "overall"


def get_risk_info(prob, threshold, female, rural, sc_st, illiterate):
    """Returns (risk_label, css_class, colour, recommendation, actions)"""
    positive = prob >= threshold
    prob_pct = prob * 100

    if prob_pct < 5:
        label = "Very Low Risk"
        css   = "risk-low"
        col   = "#1E8449"
        rec   = ("Predicted probability of incident CI is very low (<5%). "
                 "Routine annual wellness check is sufficient. "
                 "Maintain cognitive health through social engagement and physical activity.")
        acts  = [
            "✅ Annual cognitive wellness check",
            "📚 Encourage reading and mentally stimulating activities",
            "🏃 Promote regular physical activity",
            "🥗 Nutritional monitoring — maintain mid-arm circumference >24 cm",
            "📅 Reassess in 12 months",
        ]
    elif prob_pct < 15:
        label = "Low–Moderate Risk"
        css   = "risk-low"
        col   = "#2E7D32"
        rec   = (f"Predicted probability is {prob_pct:.1f}%. "
                 "Below the individual-specific screening threshold. "
                 "Some risk factors present — preventive intervention is worthwhile.")
        acts  = [
            "✅ Brief cognitive screening (Mini-Cog) at next visit",
            "📋 Address modifiable risk factors (depression, nutrition, inactivity)",
            "🥗 Nutritional assessment if mid-arm circumference <24 cm",
            "📅 Reassess in 6–12 months",
        ]
    elif not positive:
        label = "Moderate Risk"
        css   = "risk-moderate"
        col   = "#975A16"
        rec   = (f"Predicted probability is {prob_pct:.1f}%. "
                 "Approaching the screening threshold for this individual's subgroup. "
                 "Increased monitoring recommended.")
        acts  = [
            "⚠️ Administer Mini-Mental State Exam or MoCA within 3 months",
            "📋 Document informant concerns carefully at every visit",
            "🥗 Nutritional support if mid-arm circumference <24 cm",
            "🧠 Encourage cognitive stimulation — reading, puzzles, social engagement",
            "💊 Review medications for cognition-impairing drugs",
            "📅 Reassess ICIRS score in 6 months",
        ]
    elif prob_pct < 30:
        label = "High Risk"
        css   = "risk-high"
        col   = "#C53030"
        rec   = (f"Predicted probability is {prob_pct:.1f}%. "
                 "Above the Youden-optimal threshold for this individual's subgroup. "
                 "Formal cognitive assessment is strongly recommended.")
        acts  = [
            "🔴 Refer for formal neuropsychological evaluation or memory clinic",
            "📋 Administer MoCA / Mini-Cog immediately",
            "👨‍⚕️ Medical review — thyroid, vitamin B12, depression, anaemia",
            "🥗 Nutritional intervention if mid-arm circumference <22 cm",
            "🏥 Notify primary care physician and document in health record",
            "👨‍👩‍👧 Family counselling and caregiver support planning",
            "📅 Follow-up in 4–6 weeks",
        ]
    else:
        label = "Very High Risk"
        css   = "risk-veryhigh"
        col   = "#822727"
        rec   = (f"Predicted probability is {prob_pct:.1f}%. "
                 "This places the individual in the extreme-risk category "
                 "(comparable to Cluster 6: CI rate=14.6%, OR=6.61 vs lowest-risk cluster). "
                 "Urgent specialist referral is warranted.")
        acts  = [
            "🚨 Urgent referral to neurologist or geriatric psychiatrist",
            "📋 Comprehensive neuropsychological battery immediately",
            "🧬 Brain imaging + full blood workup (thyroid, B12, homocysteine, CBC)",
            "👨‍👩‍👧 Immediate caregiver involvement and safety planning",
            "📑 Review all medications for cognition-impairing effects",
            "🏥 Consider community mental health team or day care involvement",
            "📅 Specialist follow-up within 4 weeks",
        ]

    # Add equity-specific warning if applicable
    equity_warnings = []
    if illiterate:
        equity_warnings.append(
            "Illiterate subgroup: AUC=0.739 (lower precision). "
            "Threshold raised to 0.621 to reduce false positives."
        )
    if sc_st:
        equity_warnings.append(
            "SC/ST subgroup: AUC=0.781. Threshold adjusted to 0.594."
        )
    if female and not illiterate:
        equity_warnings.append(
            "Female subgroup: AUC=0.774 (vs male 0.895). "
            "Score accounts for female sex (+0.21 logit) directly."
        )

    return label, css, col, rec, acts, equity_warnings


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>🧠 ICIRS-Extended — Cognitive Impairment Risk Screener</h1>
  <p>
    Extended 14-variable model &nbsp;|&nbsp; AUC = 0.829 &nbsp;|&nbsp;
    Accounts for sex · rural/urban · SC/ST caste · depression · BMI · marital status &nbsp;|&nbsp;
    Subgroup-specific thresholds &nbsp;|&nbsp; LASI-DAD Wave 1–Wave 2 · IIPS Mumbai · Ramnaresh (2024–25)
  </p>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋  Risk Assessment", "ℹ️  About the Model", "📊  Reference Tables"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1.15, 0.85], gap="large")

    with col_form:

        # ── Section 1: Demographics ────────────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=60, max_value=100,
                                  value=68, step=1,
                                  help="Validated for age 60+. Each year adds +0.090 to log-odds.")
        with c2:
            sex_opt = st.selectbox("Sex", ["Male", "Female"],
                                   help="Female sex adds +0.205 to log-odds (reflects education gap, not biology).")

        c3, c4 = st.columns(2)
        with c3:
            residence = st.selectbox("Residence", ["Urban", "Rural"],
                                     help="Rural residence adds +0.152 to log-odds.")
        with c4:
            caste = st.selectbox("Social group",
                                 ["General / OBC", "Scheduled Caste (SC)", "Scheduled Tribe (ST)"],
                                 help="SC/ST adds +0.375 to log-odds — reflects cumulative social disadvantage.")

        married_opt = st.radio("Marital status",
                               ["Married / partnered", "Widowed / separated / never married"],
                               horizontal=True,
                               help="Being married is protective (β=−0.155). "
                                    "Cluster 5 (Very high-risk) had 0% married and 86.7% female.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 2: Education & Literacy ────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📚 Education & Literacy</div>',
                    unsafe_allow_html=True)

        illiterate_opt = st.radio(
            "Can this person read?",
            ["Yes — can read", "No — cannot read (illiterate)"],
            help="Strongest predictor: β=+0.836. Illiterate individuals have ~7× higher crude CI risk (LASI-DAD)."
        )
        illiterate_bool = illiterate_opt.startswith("No")

        educ_yrs = st.slider("Years of formal education completed",
                             min_value=0, max_value=20, value=5, step=1,
                             help="Each additional year is protective (β=−0.132). "
                                  "Average in incident CI group: 1.16 yrs vs 4.56 in intact group.")

        reads_opt = st.radio("Reads regularly? (books, newspapers, phone, etc.)",
                             ["Yes — reads regularly", "No — does not read"],
                             horizontal=True,
                             help="Protective (β=−0.066). All members of the Lower-risk cluster read regularly.")
        reads_bool = reads_opt.startswith("Yes")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 3: Functional Status ───────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🦽 Functional Status (IADL)</div>',
                    unsafe_allow_html=True)

        st.caption("Count the number of the following where the person needs **help or cannot do alone**: "
                   "preparing food · managing money · using transport · shopping · "
                   "taking medications · using telephone · housework")
        iadl = st.slider("IADL limitations (0 = fully independent, 7 = dependent in all)",
                         min_value=0, max_value=7, value=1, step=1,
                         help="Each limitation adds +0.203 to log-odds. "
                              "Adjusted OR=1.239 per limitation (FDR-significant).")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 4: Nutrition ───────────────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🥗 Nutritional Status</div>',
                    unsafe_allow_html=True)

        c5, c6 = st.columns(2)
        with c5:
            mac_avail = st.radio("Mid-arm circumference measured?",
                                 ["Yes", "No — skip"], horizontal=True)
        with c6:
            if mac_avail == "Yes":
                midarm_cm = st.number_input("Mid-arm circumference (cm)",
                                            min_value=15.0, max_value=40.0,
                                            value=26.0, step=0.5,
                                            help=">26 cm = adequate. Each cm is protective (β=−0.105).")
            else:
                midarm_cm = 26.5
                st.info("Using population mean (26.5 cm) — measure with tape for accuracy.")

        bmi_avail = st.radio("BMI available?", ["Yes", "No — skip"], horizontal=True)
        if bmi_avail == "Yes":
            bmi = st.slider("BMI (kg/m²)", min_value=12.0, max_value=45.0, value=23.0, step=0.5,
                            help="BMI has very weak effect (β=−0.005) — included for completeness.")
        else:
            bmi = 23.2  # population mean

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 5: Mental Health & Lifestyle ───────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧘 Mental Health & Lifestyle</div>',
                    unsafe_allow_html=True)

        dep_avail = st.radio("CESD-10 depression score available?",
                             ["Yes", "No — skip"], horizontal=True,
                             help="10-item depression scale (0–30). Score >10 = depressive symptoms.")
        if dep_avail == "Yes":
            depression = st.slider("CESD-10 score (0 = no symptoms, 30 = severe)",
                                   min_value=0, max_value=30, value=8, step=1,
                                   help="Note: depression coefficient is near zero (β=−0.025) and "
                                        "not FDR-significant. Included as it adds marginal information.")
        else:
            depression = 9.1  # population mean

        exercise_opt = st.radio("Does this person exercise or do any sport regularly?",
                                ["Yes — exercises regularly", "No — sedentary"],
                                horizontal=True,
                                help="Exercise coefficient is very small (β=+0.031) and not significant. "
                                     "Included for completeness — protective effect likely mediated by "
                                     "cognitive reserve pathway already captured by education.")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 6: Informant Memory ────────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👨‍👩‍👧 Informant Memory Assessment</div>',
                    unsafe_allow_html=True)

        st.markdown("Ask a **family member or close companion** who knows the person well:")
        st.markdown(
            "> *\"Have you noticed any memory problems or changes in this person "
            "over the past year — forgetting recent events, getting confused, "
            "or worse memory than others their age?\"*"
        )
        infmem_opt = st.radio("Informant's response:",
                              ["No — no memory concerns", "Yes — informant reports memory concern"],
                              help="Strongest non-cognitive predictor: β=+0.579, aOR=2.142 (FDR p=0.010). "
                                   "100% of the Extreme-risk GMM cluster (CI rate=14.6%) had this flag. "
                                   "This single question captures early pre-clinical change noticed by family.")
        infmem_bool = infmem_opt.startswith("Yes")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: Results ──────────────────────────────────────────────────────────
    with col_result:

        # Parse inputs
        female_bool   = sex_opt == "Female"
        rural_bool    = residence == "Rural"
        sc_st_bool    = caste in ["Scheduled Caste (SC)", "Scheduled Tribe (ST)"]
        married_bool  = married_opt.startswith("Married")
        exercise_bool = exercise_opt.startswith("Yes")

        # Compute probability
        prob = compute_probability(
            age=age, illiterate=illiterate_bool, educ_yrs=educ_yrs,
            iadl=iadl, infmem=infmem_bool, midarm_cm=midarm_cm,
            reads=reads_bool, female=female_bool, rural=rural_bool,
            sc_st=sc_st_bool, depression=depression, bmi=bmi,
            exercise=exercise_bool, married=married_bool,
        )

        # Get subgroup-specific threshold
        sg_key     = get_subgroup_key(female_bool, rural_bool, illiterate_bool, sc_st_bool)
        threshold  = THRESHOLDS[sg_key]

        # Get risk category
        label, css_class, colour, rec_text, actions, equity_notes = get_risk_info(
            prob=prob, threshold=threshold,
            female=female_bool, rural=rural_bool,
            sc_st=sc_st_bool, illiterate=illiterate_bool,
        )

        # Risk display
        st.markdown(f"""
        <div class="score-box {css_class}">
          <div class="score-number">{prob*100:.1f}%</div>
          <div class="score-label">predicted probability of incident CI</div>
          <div class="risk-badge"
               style="background:{colour}22;color:{colour};border:1.5px solid {colour}aa;">
            {label}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge bar
        bar_colours = {
            "risk-low":"#48BB78","risk-moderate":"#F6AD55",
            "risk-high":"#FC8181","risk-veryhigh":"#E53E3E"
        }
        bar_col = bar_colours.get(css_class, "#4299E1")
        bar_pct = min(prob * 100 * 2.5, 100)  # scale: 40% prob = full bar

        st.markdown(f"""
        <div class="prob-gauge">
          <div class="prob-fill" style="width:{bar_pct:.1f}%;background:{bar_col};">
            {prob*100:.1f}%
          </div>
        </div>
        <div style="font-size:0.73rem;color:#718096;display:flex;justify-content:space-between;">
          <span>0%</span>
          <span>Threshold: {threshold*100:.1f}%</span>
          <span>40%+</span>
        </div>
        """, unsafe_allow_html=True)

        # Screen result banner
        positive_screen = prob >= threshold
        if positive_screen:
            st.markdown(f"""
            <div style="margin-top:0.5rem;padding:0.5rem 0.8rem;background:#FFF5F5;
                 border:1px solid #FC8181;border-radius:8px;font-size:0.82rem;color:#C53030;">
              ⚠️ <strong>POSITIVE SCREEN</strong> — Probability {prob*100:.1f}% ≥
              subgroup threshold {threshold*100:.1f}%<br>
              Subgroup: <em>{sg_key.replace('_',' ').title()}</em>
              (AUC={SUBGROUP_AUC.get(sg_key.replace('_',' ').title(), 0.829):.3f})
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="margin-top:0.5rem;padding:0.5rem 0.8rem;background:#F0FFF4;
                 border:1px solid #48BB78;border-radius:8px;font-size:0.82rem;color:#1E8449;">
              ✅ <strong>NEGATIVE SCREEN</strong> — Probability {prob*100:.1f}% &lt;
              subgroup threshold {threshold*100:.1f}%<br>
              NPV ~98% — low probability of developing CI within 3–4 years
            </div>
            """, unsafe_allow_html=True)

        # ── Variable contributions ──────────────────────────────────────────────
        st.markdown('<div class="section-card" style="margin-top:0.9rem;">',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 Probability Drivers</div>',
                    unsafe_allow_html=True)

        # Each variable's contribution to the logit
        contributions = [
            ("Age",               MODEL["age"]        * age,                         True),
            ("Illiteracy",        MODEL["illiterate"] * (1 if illiterate_bool else 0), True),
            ("Education (yrs)",   MODEL["educ_yrs"]   * educ_yrs,                    False),
            ("IADL limitations",  MODEL["iadl"]       * iadl,                        True),
            ("Informant memory",  MODEL["infmem"]     * (1 if infmem_bool else 0),   True),
            ("Mid-arm circ.",     MODEL["midarm"]     * midarm_cm,                   False),
            ("Reading habit",     MODEL["reading"]    * (1 if reads_bool else 0),    False),
            ("Female sex",        MODEL["female"]     * (1 if female_bool else 0),   True),
            ("Rural residence",   MODEL["rural"]      * (1 if rural_bool else 0),    True),
            ("SC/ST caste",       MODEL["sc_st"]      * (1 if sc_st_bool else 0),    True),
            ("Depression",        MODEL["depression"] * depression,                  None),
            ("BMI",               MODEL["bmi"]        * bmi,                         None),
            ("Exercise",          MODEL["exercise"]   * (1 if exercise_bool else 0), None),
            ("Married",           MODEL["married"]    * (1 if married_bool else 0),  False),
        ]

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        for name, contrib, is_risk in contributions:
            if abs(contrib) < 0.001:
                continue
            direction = "▲ risk" if contrib > 0 else "▼ protect"
            dir_col   = "#E53E3E" if contrib > 0 else "#48BB78"
            bar_w     = min(abs(contrib) / 2.0 * 100, 100)
            bar_c     = "#FED7D7" if contrib > 0 else "#C6F6D5"
            bar_fc    = "#FC8181" if contrib > 0 else "#48BB78"
            st.markdown(f"""
            <div class="item-row">
              <div style="width:45%;font-size:0.83rem;">{name}</div>
              <div style="width:35%;">
                <div style="background:{bar_c};border-radius:6px;height:8px;overflow:hidden;">
                  <div style="width:{bar_w:.0f}%;height:100%;background:{bar_fc};border-radius:6px;">
                  </div>
                </div>
              </div>
              <div style="width:20%;text-align:right;font-size:0.78rem;
                   color:{dir_col};font-weight:600;">
                {contrib:+.3f}<br>{direction}
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Equity notes ────────────────────────────────────────────────────────
        if equity_notes:
            st.markdown('<div class="warn-note">', unsafe_allow_html=True)
            st.markdown("**📌 Equity note for this individual's profile:**")
            for note in equity_notes:
                st.markdown(f"• {note}")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Clinical Recommendation ─────────────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💊 Clinical Recommendation</div>',
                    unsafe_allow_html=True)
        rec_css = css_class.replace("risk-", "rec-")
        st.markdown(f'<div class="rec-box {rec_css}">{rec_text}</div>',
                    unsafe_allow_html=True)
        st.markdown("**Action checklist:**")
        for action in actions:
            st.markdown(f"- {action}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-note">
          <strong>⚕️ Disclaimer:</strong> ICIRS-Extended is a <em>community screening tool</em>.
          A positive screen does not confirm cognitive impairment or dementia.
          All screen-positive individuals must be referred for formal clinical assessment.
          This tool does not replace clinical judgement.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    colA, colB = st.columns(2, gap="large")

    with colA:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔬 Why Extended?</div>', unsafe_allow_html=True)
        st.markdown("""
The original 7-variable ICIRS (integer score, AUC=0.821) did **not directly account
for sex, rural/urban residence, SC/ST caste, depression, BMI, or marital status**.
This version corrects that.

**Key finding from the extended analysis:**

Adding these variables to the logistic regression model raises AUC from **0.821 →
0.829** — a modest improvement at the overall level, but with important
implications for individual-level prediction:

| Variable | β coefficient | Interpretation |
|----------|:-------------:|----------------|
| Female sex | +0.205 | Women carry higher CI risk, mediated by the education gap |
| Rural residence | +0.152 | Rural elderly face structural barriers (less healthcare access, lower literacy) |
| SC/ST caste | +0.375 | Cumulative social disadvantage raises risk independently |
| Married | −0.155 | Social support is protective |
| Depression | −0.025 | Near-zero after controlling for other variables |
| BMI | −0.005 | Very weak (malnutrition captured better by mid-arm circ.) |
| Exercise | +0.031 | Non-significant |

**Why not more improvement?** Sex, rural status, and caste are **mediated** by
illiteracy and education — once those are in the model, the additional
marginal information from the mediators is small. This is the correct
epidemiological finding, not a limitation of the approach.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Model Comparison</div>',
                    unsafe_allow_html=True)
        st.markdown("""
| Model | Variables | AUC |
|-------|-----------|-----|
| CAIDE (Western) | 4/6 items | 0.718 |
| ANU-ADRI (Western) | 8/13 items | 0.740 |
| ICIRS v1 (integer score) | 7 vars | 0.821 |
| **ICIRS-Extended (this app)** | **14 vars** | **0.829** |

**Subgroup AUCs (Extended model):**

| Subgroup | AUC |
|----------|-----|
| Male | 0.895 ✅ |
| Female | 0.774 ⚠️ |
| Urban | 0.859 ✅ |
| Rural | 0.798 |
| Literate | 0.852 ✅ |
| Illiterate | 0.739 ⚠️ |
| General/OBC | 0.841 ✅ |
| SC/ST | 0.781 ⚠️ |

⚠️ = AUC meaningfully below overall. Subgroup-specific Youden thresholds
are applied automatically for these groups.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📐 Subgroup-Specific Thresholds</div>',
                    unsafe_allow_html=True)
        st.markdown("""
The app automatically selects the **Youden-optimal probability threshold**
based on this individual's subgroup profile. This is the key innovation
that makes the extended model more equitable than a single fixed cut-point.

| Subgroup | Threshold | Sensitivity | Specificity |
|----------|:---------:|:-----------:|:-----------:|
| Overall | 49.6% | 81.8% | 72.2% |
| Male | 40.5% | 87.0% | 78.6% |
| Female | 50.5% | 83.3% | 62.0% |
| Urban | 30.1% | 90.9% | 69.0% |
| Rural | 52.5% | 84.3% | 64.6% |
| Literate | 26.8% | 72.2% | 85.2% |
| **Illiterate** | **62.1%** | 70.0% | 67.7% |
| General/OBC | 49.6% | 83.0% | 74.7% |
| **SC/ST** | **59.4%** | 81.2% | 66.3% |

The illiterate threshold is **raised** to 62.1% (vs 49.6% overall) because
illiteracy itself contributes so strongly to the score that a lower threshold
would flag almost all illiterate elderly — creating unworkable false positive rates.

**Priority for threshold selection:**
Illiterate > SC/ST > Female+Rural > Female > Rural > Overall
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📖 Data Source & Validation</div>',
                    unsafe_allow_html=True)
        st.markdown("""
**Dataset:** LASI-DAD (Longitudinal Ageing Study in India —
Diagnostic Assessment of Dementia)

- Wave 1: 2017–18 (N=4,096 enrolled)
- Wave 2: 2020–21 (N=3,236 followed)
- Analytic cohort: **N=1,869** cognitively intact at Wave 1
- Incident CI cases: **148 (7.9%)** at Wave 2

**Model training:**
- Logistic regression, class_weight=balanced, C=0.5
- MICE imputation (m=5, BayesianRidge)
- Survey weights from r1labwgt

**Internal validation:**
- 500-resample bootstrap optimism correction
- Optimism-corrected AUC: **0.812** (original integer ICIRS)
- Hosmer-Lemeshow p=0.598 (good calibration)
- Brier score = 0.063

**Limitation:** No external validation cohort available in India.
Performance may vary in populations with different baseline risk profiles.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    colR1, colR2 = st.columns(2, gap="large")

    with colR1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Field Screening Card</div>',
                    unsafe_allow_html=True)
        st.markdown("""
*For ASHA workers and community health volunteers*

**Step 1 — Record details (5 items, always asked):**

| Item | How to assess | Contribution |
|------|--------------|-------------|
| Age | Ask date of birth | +0.090 per year |
| Can read? | Ask to read any text | +0.836 if cannot |
| Years of school | Ask education history | −0.132 per year |
| IADL count | Ask 7 daily activities | +0.203 per item |
| Informant memory | Ask family member | +0.579 if yes |

**Step 2 — Record if available (5 items):**

| Item | How to assess | Contribution |
|------|--------------|-------------|
| Sex | Observe / ask | +0.205 if female |
| Rural / urban | Ask area | +0.152 if rural |
| Caste | Ask social group | +0.375 if SC/ST |
| Reads regularly | Ask daily habit | −0.066 if reads |
| Mid-arm (cm) | Tape measure | −0.105 per cm |
| Married | Ask | −0.155 if married |

**Step 3 — Enter values in the app and read the result**

Screen positive? → Refer to primary health centre
Screen negative? → Reassess in 12 months
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📏 IADL Assessment Guide</div>',
                    unsafe_allow_html=True)
        st.markdown("""
Ask: *"Can you do this alone without help?"*

1. **Prepare food** — cook a simple meal
2. **Manage money** — count change, pay bills
3. **Use transport** — take bus or auto alone
4. **Go shopping** — buy groceries independently
5. **Take medications** — correct dose at correct time
6. **Use telephone** — dial numbers, use mobile
7. **Do housework** — sweep, mop, basic cleaning

**Score:** Count the number where the person
**cannot do alone or needs help**.

0 = fully independent → low risk
3+ = significant limitations → refer
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with colR2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧬 GMM Risk Cluster Profiles</div>',
                    unsafe_allow_html=True)
        st.markdown("""
Six latent risk profiles from LASI-DAD GMM clustering (k=6):

| Cluster | n | CI Rate | Key characteristics |
|---------|---|---------|---------------------|
| 1 Lower-risk | 373 | **2.1%** | High educ (8.8yr), 100% read, low IADL (0.5), 12% illiterate |
| 2 Moderate-risk | 251 | 2.8% | High educ (7.9yr), 100% read, OR=1.05 vs C1 (ns) |
| 3 Intermediate | 389 | 8.7% | Illiterate (70%), married (100%), 0% read, OR=5.97*** |
| 4 Higher-risk | 337 | 9.2% | Illiterate (79%), married (100%), 0% read, 68% female |
| 5 Very high | 361 | 12.5% | 85% illiterate, 0% married, **87% female**, 0% read |
| 6 Extreme | 158 | **14.6%** | **100% informant memory concern**, OR=6.61*** |

**Compare your individual's profile to these clusters
to contextualise the model output.**

Cluster 6 key insight: The *only* distinguishing
feature of the highest-risk cluster is universal
informant memory concern — this single question
captures early pre-clinical changes not detectable
by objective cognitive tests.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📏 Mid-Arm Measurement Guide</div>',
                    unsafe_allow_html=True)
        st.markdown("""
**Technique:**
1. Person stands relaxed, arm at side
2. Find midpoint between shoulder tip and elbow
3. Wrap tape horizontally — snug but not compressing
4. Read to nearest 0.5 cm

**Reference values:**
| MAC | Status | Risk |
|-----|--------|------|
| >26 cm | Adequate nutrition | Baseline |
| 24–26 cm | Mild malnutrition risk | Moderate |
| 22–24 cm | Moderate malnutrition | Higher |
| <22 cm | Severe protein-energy malnutrition | Highest |

The malnutrition–CI pathway operates through:
reduced cerebral perfusion, micronutrient
deficiencies, and immune compromise.
Mid-arm circumference: β=−0.105 per cm.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📞 Referral Resources</div>',
                    unsafe_allow_html=True)
        st.markdown("""
**National programmes:**
- **NPHCE** (National Programme for Health Care of the Elderly)
  — geriatric OPD at district hospitals
- **IPOP** (Integrated Programme for Older Persons)
  — community-based elderly care
- **DEMENTIA INDIA ALLIANCE** — helpline and memory clinics

**Clinical assessment tools for referral centres:**
- Mini-Mental State Examination (MMSE)
- Montreal Cognitive Assessment (MoCA)
- Clinical Dementia Rating (CDR)
- Informant Questionnaire on Cognitive Decline (IQCODE)

**After positive ICIRS screen:**
PHC/CHC → District hospital geriatric OPD
→ Tertiary memory clinic (if warranted)
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <strong>ICIRS-Extended</strong> — India-specific Cognitive Impairment Risk Score v2.0
  &nbsp;|&nbsp; 14-variable logistic regression model (AUC=0.829)
  &nbsp;|&nbsp; Subgroup-specific Youden thresholds
  &nbsp;|&nbsp; Accounts for sex · rural/urban · SC/ST caste · depression · BMI · marital status<br>
  Developed by <strong>Ramnaresh</strong>,
  M.Sc. Survey Research &amp; Data Analytics,
  International Institute for Population Sciences, Mumbai, 2024–25
  &nbsp;|&nbsp; LASI-DAD Wave 1–Wave 2 (N=1,869)<br>
  <em>For community screening purposes only.
  Not a diagnostic instrument.
  Always refer screen-positive individuals for formal clinical assessment.</em>
</div>
""", unsafe_allow_html=True)
