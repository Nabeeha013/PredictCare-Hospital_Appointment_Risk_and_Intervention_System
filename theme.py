"""PredictCare dashboard theme — colors and CSS."""

COLORS = {
    "navy": "#1B2A4A",
    "navy_light": "#243656",
    "teal": "#0891B2",
    "teal_light": "#22D3EE",
    "bg": "#E8F4FC",
    "bg_card": "#FFFFFF",
    "blue": "#4A90D9",
    "coral": "#F4847A",
    "green": "#22C55E",
    "yellow": "#EAB308",
    "orange": "#F97316",
    "red": "#EF4444",
    "text": "#1E293B",
    "text_muted": "#64748B",
    "border": "#CBD5E1",
    "insight_bg": "#DBEAFE",
}

PAGES = [
    ("Dashboard", "📊"),
    ("Single Patient", "👤"),
    ("Hospital Entry", "🏥"),
    ("Batch Upload", "📂"),
    ("Model Performance", "📈"),
    ("About", "ℹ️"),
]

NEW_BADGE_PAGES = {"Hospital Entry"}


def inject_css():
    c = COLORS
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: linear-gradient(180deg, {c['bg']} 0%, #F0F9FF 100%);
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}

.block-container {{
    padding-top: 1.5rem;
    max-width: 1400px;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {c['navy']} 0%, #152238 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}}

section[data-testid="stSidebar"] > div {{
    background: transparent;
}}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label {{
    color: #E2E8F0 !important;
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    background: transparent;
    border-radius: 10px;
    padding: 0.55rem 0.75rem;
    margin: 0.15rem 0;
    width: 100%;
    transition: background 0.2s;
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background: rgba(255,255,255,0.08);
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
    background: rgba(74, 144, 217, 0.35) !important;
    border-left: 3px solid {c['teal_light']};
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {{
    gap: 0.25rem;
}}

.kpi-card {{
    background: {c['bg_card']};
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 2px 14px rgba(27, 42, 74, 0.08);
    border: 1px solid rgba(203, 213, 225, 0.5);
    height: 100%;
}}

.kpi-label {{
    color: {c['text_muted']};
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 0.35rem;
}}

.kpi-value {{
    color: {c['navy']};
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1.2;
}}

.kpi-icon {{
    font-size: 1.4rem;
    float: right;
    opacity: 0.85;
}}

.chart-card {{
    background: {c['bg_card']};
    border-radius: 14px;
    padding: 1rem 1rem 0.5rem 1rem;
    box-shadow: 0 2px 14px rgba(27, 42, 74, 0.08);
    border: 1px solid rgba(203, 213, 225, 0.45);
    height: 100%;
}}

.chart-title {{
    color: {c['navy']};
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}}

.insight-banner {{
    background: {c['insight_bg']};
    border-left: 4px solid {c['blue']};
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    color: {c['text']};
    font-size: 0.92rem;
    margin: 1rem 0;
}}

.info-card {{
    background: {c['bg_card']};
    border-radius: 14px;
    padding: 1.25rem;
    box-shadow: 0 2px 14px rgba(27, 42, 74, 0.08);
    border: 1px solid rgba(203, 213, 225, 0.45);
    height: 100%;
}}

.info-card h3 {{
    color: {c['navy']};
    font-size: 1rem;
    font-weight: 700;
    margin-top: 0;
}}

.risk-row {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.88rem;
}}

.risk-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}}

.badge-new {{
    background: {c['teal']};
    color: white;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 0.12rem 0.4rem;
    border-radius: 999px;
    margin-left: 0.35rem;
    vertical-align: middle;
}}

.hero-tagline {{
    color: {c['teal']};
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
}}

.hero-subtitle {{
    color: {c['text_muted']};
    font-size: 0.88rem;
    margin-top: 0.25rem;
}}

.sidebar-footer {{
    color: rgba(226, 232, 240, 0.75);
    font-size: 0.78rem;
    font-style: italic;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.1);
    line-height: 1.5;
}}

.stDownloadButton button {{
    border-radius: 10px;
    border-color: {c['teal']};
    color: {c['teal']};
}}

div[data-testid="stFormSubmitButton"] button,
.stButton > button[kind="primary"] {{
    background: linear-gradient(90deg, {c['teal']} 0%, {c['blue']} 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}}

.success-box {{
    background: #ECFDF5;
    border-left: 4px solid {c['green']};
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-size: 0.9rem;
}}
</style>
"""


def kpi_card(label, value, icon=""):
    return f"""
<div class="kpi-card">
    <span class="kpi-icon">{icon}</span>
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
</div>
"""


def chart_card(title, fig, height=320):
    import streamlit as st

    st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def risk_level_html(level, range_text, intervention, color):
    return f"""
<div class="risk-row">
    <span class="risk-dot" style="background:{color};"></span>
    <strong>{level}</strong>
    <span style="color:#64748B;">({range_text})</span>
    <span style="margin-left:auto;color:#475569;">{intervention}</span>
</div>
"""
