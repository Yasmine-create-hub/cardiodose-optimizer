# =============================================================================
#  CardioDose Optimizer  v6.0 — Version Finale Ultime
#  Aide à la décision dosimétrique — Cardiologie Interventionnelle — CHU Algérie
#  TP de fin d'études — Recherche Opérationnelle / Data Science médicale
# =============================================================================
#  Fonctionnalités :
#   • Chargement CSV ultra-robuste (détection auto encodage/séparateur + upload)
#   • Random Forest (4 cibles dosimétriques) + IsolationForest (détection atypiques)
#   • K-Means clustering (n=4) — figé sur les 160 cas réels
#   • Silhouette Score + Méthode Elbow + PCA 2D & 3D interactives
#   • Heatmap de corrélations (Spearman)
#   • Boxplots + Violin plots par variable et par procédure
#   • SHAP complètement optionnel (try/except élégant)
#   • Rapport PDF professionnel et complet (ReportLab optionnel)
#   • Export CSV & Excel (openpyxl optionnel)
#   • UX soignée : spinner, messages colorés, bouton Nouvelle Prédiction
#   • Code modulaire, commenté, 100 % stable pour une soutenance
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import base64
import warnings

warnings.filterwarnings("ignore")

# ─── CONFIG PAGE ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioDose Optimizer  — CHU",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# LOGO OFFICIEL — CHU MUSTAPHA BACHA, ALGER
# ══════════════════════════════════════════════════════════════════════════════
# Logo institutionnel encodé en base64 (PNG), utilisé à la fois dans le
# dashboard (header / sidebar) et dans l'en-tête du rapport PDF.
LOGO_CHU_B64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAABhGlDQ1BJQ0MgcHJvZmlsZQAAeJx9kT1Iw1AUhU9T"
        "pVorHewg4pChOlkQFXHUKhShQqgVWnUweekfNGlIUlwcBdeCgz+LVQcXZ10dXAVB8AfE2cFJ0UVKvK8ptIjxweV9"
        "nPfO4b77AKFeZprVNQ5oum2mEnExk10VA68IIow+ql6ZWcacJCXhub7u4eP7XYxned/7c/WrOYsBPpF4lhmmTbxB"
        "PL1pG5z3iSOsKKvE58RjJjVI/Mh1xeU3zoUmCzwzYqZT88QRYrHQwUoHs6KpEU8RR1VNp3wh47LKeYuzVq6yVp/8"
        "haGcvrLMdaphJLCIJUgQoaCKEsqwEaNdJ8VCis7jHv6hpl8il0KuEhg5FlCBBrnpB/+D37O18pMTblIoDnS/OM7H"
        "CBDYBRo1x/k+dpzGCeB/Bq70tr9SB2Y+Sa+1tegREN4GLq7bmrIHXO4Ag0+GbMpNyU8l5PPA+xl9UxYYuAWCa+7c"
        "Wuc4fQDSNKvkDXBwCIwWKHvd4909nXP7905rfj8Ke3J9Yn7o7QAADXhpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAA"
        "ADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+Cjx4OnhtcG1ldGEg"
        "eG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDQuNC4wLUV4aXYyIj4KIDxyZGY6UkRG"
        "IHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgPHJkZjpE"
        "ZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgIHhtbG5zOnhtcE1NPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8x"
        "LjAvbW0vIgogICAgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJj"
        "ZUV2ZW50IyIKICAgIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIKICAgIHhtbG5z"
        "OkdJTVA9Imh0dHA6Ly93d3cuZ2ltcC5vcmcveG1wLyIKICAgIHhtbG5zOnRpZmY9Imh0dHA6Ly9ucy5hZG9iZS5j"
        "b20vdGlmZi8xLjAvIgogICAgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIgogICB4bXBN"
        "TTpEb2N1bWVudElEPSJnaW1wOmRvY2lkOmdpbXA6YTJkZWYzYmEtMWNjMS00MWQ1LWIwYjYtNDUxZDIyYjJjZDM3"
        "IgogICB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOjRmZWE5MzExLWZjMzQtNDE5ZS04MTYyLWQxM2FhODZjNWQ2"
        "ZCIKICAgeG1wTU06T3JpZ2luYWxEb2N1bWVudElEPSJ4bXAuZGlkOjY2NzVlNGRjLWE2ZWItNGU3Mi1hZWNlLWIw"
        "NzVkN2ZkMjQ1OSIKICAgZGM6Rm9ybWF0PSJpbWFnZS9wbmciCiAgIEdJTVA6QVBJPSIyLjAiCiAgIEdJTVA6UGxh"
        "dGZvcm09IkxpbnV4IgogICBHSU1QOlRpbWVTdGFtcD0iMTcyODk5MDk0MzYyMTI1MiIKICAgR0lNUDpWZXJzaW9u"
        "PSIyLjEwLjM4IgogICB0aWZmOk9yaWVudGF0aW9uPSIxIgogICB4bXA6Q3JlYXRvclRvb2w9IkdJTVAgMi4xMCIK"
        "ICAgeG1wOk1ldGFkYXRhRGF0ZT0iMjAyNDoxMDoxNVQxMjoxNTozNCswMTowMCIKICAgeG1wOk1vZGlmeURhdGU9"
        "IjIwMjQ6MTA6MTVUMTI6MTU6MzQrMDE6MDAiPgogICA8eG1wTU06SGlzdG9yeT4KICAgIDxyZGY6U2VxPgogICAg"
        "IDxyZGY6bGkKICAgICAgc3RFdnQ6YWN0aW9uPSJzYXZlZCIKICAgICAgc3RFdnQ6Y2hhbmdlZD0iLyIKICAgICAg"
        "c3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDphYTBiYWMwYy0zYTc1LTQ3ZWUtYTg5My1hOGNhNWFkNmU2YzkiCiAg"
        "ICAgIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkdpbXAgMi4xMCAoTGludXgpIgogICAgICBzdEV2dDp3aGVuPSIyMDI0"
        "LTEwLTE1VDEyOjE1OjQzKzAxOjAwIi8+CiAgICA8L3JkZjpTZXE+CiAgIDwveG1wTU06SGlzdG9yeT4KICA8L3Jk"
        "ZjpEZXNjcmlwdGlvbj4KIDwvcmRmOlJERj4KPC94OnhtcG1ldGE+CiAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "IAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
        "ICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAKPD94cGFja2V0IGVuZD0idyI/PmUIDDoA"
        "AAAGYktHRAD/AP8A/6C9p5MAAAAJcEhZcwAATOUAAEzlAXXO8JUAABJ4SURBVHja7Z15fFzVdce/YxsIkqEEJJYB"
        "JChlpAfGmLFpU/SE2xK8lCRgm0KhkJISbFxCsLNArs0SwPYlEBCFBrAJNIChIRQTkgJmMSnWFTQBDwaMrz1ZkAQd"
        "gi12LDDGmv5xz+DxMPu8kWRF5/PRR7O8ee+++7tnP/c8GKZhGqZhGqYgyfNVneerRnndWODYKZ6vZsrrGs9X43ek"
        "ex2xA431/LET54WAaQWOs8CqFD7A54YBCZis0T3A2y8+tSgJ1BQ4tgs4Q96GgI+ycNHlw4DkFjEn5frusFY1Ku3t"
        "2/L/rRznaU172y3/j0p7TZrIGzEMSG4a7fnKS5uwb3q+Wuz56qVkkic8X+3u+epo4Eg5ZGUWMBqBQ+X1GcB0z1d1"
        "QBdwheeraWm6ZxrwpuergzLOcZjnq1P+5AGxRi/N0As/Fj1wHfAEcA1wjLzHGr02y2mSwJdEgW8EnpNjHwP+DRgF"
        "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"
)


def get_logo_bytes() -> bytes:
    """Décode le logo CHU une seule fois (mis en cache) et retourne les bytes PNG."""
    return base64.b64decode(LOGO_CHU_B64)


@st.cache_data(show_spinner=False)
def get_logo_data_uri() -> str:
    """Retourne le logo CHU au format data-URI, prêt à être injecté dans du HTML/CSS."""
    return "data:image/png;base64," + LOGO_CHU_B64

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CSS GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem !important; }

/* ── HEADER ── */
.main-header {
    margin-top: 4px; min-height: 92px;
    background: linear-gradient(135deg, #060f1e 0%, #0d2137 50%, #060f1e 100%);
    border: 1px solid #1e3a5f; border-radius: 14px;
    padding: 20px 28px; margin-bottom: 20px; overflow: visible;
    display: flex; align-items: center; gap: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}

/* ── KPI CARDS ── */
.kpi-card {
    background: linear-gradient(145deg, #0d1b2a, #0f2035);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 16px 14px; text-align: center;
    transition: transform 0.15s, box-shadow 0.15s;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: var(--kpi-color, #22d3ee);
    border-radius: 2px 2px 0 0;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
.kpi-value { font-size: 26px; font-weight: 800; margin: 6px 0 2px; letter-spacing: -0.02em; }
.kpi-label { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600; }
.kpi-sub   { font-size: 10px; color: #334155; margin-top: 3px; }

/* ── BADGES DE RISQUE ── */
.risk-badge-A { background: #052e16; color: #4ade80; border: 3px solid #166534; }
.risk-badge-B { background: #422006; color: #fbbf24; border: 3px solid #92400e; }
.risk-badge-C { background: #431407; color: #fb923c; border: 3px solid #9a3412; }
.risk-badge-D { background: #450a0a; color: #f87171; border: 3px solid #991b1b; }
.risk-badge {
    display: inline-block; padding: 16px 30px; border-radius: 16px;
    font-size: 26px; font-weight: 800; letter-spacing: 0.02em; line-height: 1.3;
}

/* ── ALERTES ── */
.alert-critique { background:#450a0a; border-left:4px solid #f87171; padding:10px 14px; border-radius:6px; margin:5px 0; }
.alert-danger   { background:#431407; border-left:4px solid #fb923c; padding:10px 14px; border-radius:6px; margin:5px 0; }
.alert-warning  { background:#422006; border-left:4px solid #fbbf24; padding:10px 14px; border-radius:6px; margin:5px 0; }
.alert-ok       { background:#052e16; border-left:4px solid #4ade80; padding:10px 14px; border-radius:6px; margin:5px 0; }
.alert-info     { background:#172554; border-left:4px solid #60a5fa; padding:10px 14px; border-radius:6px; margin:5px 0; }

/* ── TITRES DE SECTION ── */
.section-title {
    font-size: 13px; font-weight: 700; color: #cbd5e1;
    border-bottom: 1px solid #1e3a5f; padding-bottom: 8px; margin-bottom: 16px;
    text-transform: uppercase; letter-spacing: 0.06em;
}

/* ── EXPLAIN BOX ── */
.explain-box { background:#080f1a; border:1px solid #1e3a5f; border-radius:10px; padding:16px 18px; margin-top:8px; }
.explain-box h4 { color:#22d3ee; font-size:11px; margin:0 0 10px; text-transform:uppercase; letter-spacing:0.1em; font-weight:700; }
.explain-item { font-size:12px; color:#94a3b8; padding:6px 0; border-bottom:1px solid #0f1f33; line-height:1.5; }
.explain-item:last-child { border-bottom: none; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] { background:#060e1b; border-right:1px solid #1e293b; }

/* ── TABS ── */
.stTabs { position:sticky; top:0; z-index:1000; background:#060e1b; padding-top:4px; }
.stTabs [data-baseweb="tab-list"] {
    gap:4px; background:#060e1b; border-radius:10px; padding:4px;
    border:1px solid #1e3a5f; position:sticky; top:0; z-index:1001;
}
.stTabs [data-baseweb="tab-list"] > div { background:#060e1b; }
.stTabs [data-baseweb="tab"] {
    background:transparent; color:#64748b; border-radius:7px;
    padding:9px 16px; font-size:13px; font-weight:600;
}
.stTabs [aria-selected="true"] { background:#1e3a5f !important; color:#22d3ee !important; font-weight:700; }

/* ── BADGES CLUSTER ── */
.cluster-0 { background:#052e16; color:#4ade80; border:1px solid #166534; border-radius:8px; padding:4px 12px; font-size:13px; font-weight:700; }
.cluster-1 { background:#172554; color:#60a5fa; border:1px solid #1d4ed8; border-radius:8px; padding:4px 12px; font-size:13px; font-weight:700; }
.cluster-2 { background:#422006; color:#fbbf24; border:1px solid #92400e; border-radius:8px; padding:4px 12px; font-size:13px; font-weight:700; }
.cluster-3 { background:#450a0a; color:#f87171; border:1px solid #991b1b; border-radius:8px; padding:4px 12px; font-size:13px; font-weight:700; }

/* ── METRIC BOX ── */
.metric-box { background:#0d1b2a; border:1px solid #1e3a5f; border-radius:10px; padding:14px 18px; text-align:center; }
.atypique-badge { background:#450a0a; color:#f87171; border:2px solid #991b1b; padding:4px 12px; border-radius:6px; font-size:11px; font-weight:700; }
.normal-badge   { background:#052e16; color:#4ade80; border:2px solid #166534; padding:4px 12px; border-radius:6px; font-size:11px; font-weight:700; }

/* ── BOUTON PRÉDIRE (très visible) ── */
div[data-testid="stButton"] button[kind="primary"] {
    font-size: 22px !important; font-weight: 800 !important;
    padding: 20px 0 !important; border-radius: 14px !important;
    background: linear-gradient(135deg,#0ea5e9,#22d3ee) !important;
    color: #04111f !important; border: none !important;
    box-shadow: 0 8px 28px rgba(34,211,238,0.45) !important;
    letter-spacing: 0.03em !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 34px rgba(34,211,238,0.6) !important;
}
.cta-subtitle { text-align:center; color:#64748b; font-size:13px; margin-bottom:10px; }

/* ── NOUVELLE PRÉDICTION ── */
.new-pred-hint {
    background: #172554; border: 1px solid #1d4ed8; border-radius:10px;
    padding: 10px 16px; margin-top: 16px; font-size: 13px; color: #60a5fa;
    text-align: center;
}

/* ── POLISH CSS — hover/transition/élévation sur cartes KPI ── */
.kpi-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    cursor: default;
}
.kpi-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 10px 30px rgba(34,211,238,0.18) !important;
}
.kpi-value {
    animation: metricFadeIn 0.5s ease forwards;
}
@keyframes metricFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.metric-box {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
}

/* ── CARTES PREMIUM ALERTE ── */
.alert-premium-red {
    background: linear-gradient(135deg, #450a0a, #5c0f0f);
    border: 2px solid #f87171; border-radius: 14px;
    padding: 20px 24px; margin: 12px 0;
    box-shadow: 0 4px 24px rgba(248,113,113,0.25);
}
.alert-premium-red .alert-title {
    font-size: 16px; font-weight: 800; color: #f87171; margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}
.alert-premium-row {
    display: flex; gap: 10px; margin-bottom: 6px; align-items: flex-start;
}
.alert-premium-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }
.alert-premium-text { font-size: 13px; color: #fca5a5; line-height: 1.5; }
.alert-premium-label { font-size: 11px; color: #dc2626; text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 700; }

/* ── EXECUTIVE SUMMARY CARD ── */
.exec-summary-card {
    background: linear-gradient(135deg, #0a1628, #0d2137);
    border: 1px solid #1e3a5f; border-radius: 16px; padding: 24px;
    margin-bottom: 20px;
    transition: box-shadow 0.2s ease;
}
.exec-summary-card:hover { box-shadow: 0 8px 32px rgba(34,211,238,0.12); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONSTANTES CLINIQUES
# ══════════════════════════════════════════════════════════════════════════════

NRD = {
    "France (ASN)":  {"PDS_CORO": 45,  "PDS_ANGIO": 85,  "Kerma_CORO": 500,  "Kerma_ANGIO": 1000},
    "ESC Europe":    {"PDS_CORO": 46,  "PDS_ANGIO": 80,  "Kerma_CORO": 500,  "Kerma_ANGIO": 1000},
    "IAEA":          {"PDS_CORO": 50,  "PDS_ANGIO": 100, "Kerma_CORO": 600,  "Kerma_ANGIO": 1200},
    "Algérie":       {"PDS_CORO": 50,  "PDS_ANGIO": 90,  "Kerma_CORO": 550,  "Kerma_ANGIO": 1100},
    "Belgique":      {"PDS_CORO": 42,  "PDS_ANGIO": 78,  "Kerma_CORO": 480,  "Kerma_ANGIO":  980},
    "Royaume-Uni":   {"PDS_CORO": 44,  "PDS_ANGIO": 83,  "Kerma_CORO": 510,  "Kerma_ANGIO": 1050},
}

# Langage clinique médecin (jamais les lettres ICRP brutes à l'affichage)
CLINICAL_RISK = {
    "A": {"label": "Risque Minimal",  "desc": "Dose dans les normes — aucune action particulière requise.",
          "icon": "✅", "color": "#4ade80"},
    "B": {"label": "Risque Modéré",   "desc": "Dose légèrement élevée — surveillance standard recommandée.",
          "icon": "🟡", "color": "#fbbf24"},
    "C": {"label": "Risque Élevé",    "desc": "Dose élevée — surveillance cutanée recommandée.",
          "icon": "🟠", "color": "#fb923c"},
    "D": {"label": "Risque Critique", "desc": "Dose critique — suivi dermatologique obligatoire (effet déterministe possible).",
          "icon": "🔴", "color": "#f87171"},
}

CLUSTER_NAMES  = {0: "Faible exposition", 1: "Exposition modérée",
                  2: "Exposition élevée",  3: "Exposition critique"}
CLUSTER_COLORS = {0: "#4ade80", 1: "#60a5fa", 2: "#fbbf24", 3: "#f87171"}
CLUSTER_BG     = {0: "#052e16", 1: "#172554", 2: "#422006", 3: "#450a0a"}

RF_FEATURES    = ["AGE", "SEXE_BIN", "IMC", "DUREE_MIN", "NB_SERIES", "NB_IMAGES", "PROC_BIN"]
CLUST_FEATURES = ["IMC", "DUREE_MIN", "NB_SERIES", "NB_IMAGES", "PDS_TOTAL", "KERMA_mGy"]
TARGETS        = ["PDS_TOTAL", "KERMA_mGy", "DOSE_EFF", "DOSE_PEAU"]
MAIN_COLS      = ["N", "SOURCE", "SEXE", "AGE", "IMC", "PROC_TYPE", "DUREE_MIN",
                  "NB_SERIES", "NB_IMAGES", "PDS_TOTAL", "KERMA_mGy", "DOSE_EFF",
                  "DOSE_PEAU", "CLASSE"]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CHARGEMENT DES DONNÉES (ultra-robuste)
# ══════════════════════════════════════════════════════════════════════════════

def _build_df_from_raw(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Renomme les colonnes, nettoie les types et dérive les colonnes calculées."""
    cols_map = [
        "N", "SALLE", "SEXE", "AGE", "POIDS", "TAILLE", "IMC", "CAT_IMC",
        "DIABETE", "HTA", "CHOLESTEROL", "FUMEUR", "CARDIOPATHIE",
        "NB_IMAGES", "NB_SERIES", "FOV", "DIST_SOURCE", "FLUO_PULSEE",
        "K_SCOPIE", "K_GRAPHIE", "DUREE_FLUO", "KERMA_mGy", "KERMA_Gy",
        "PDS_SCOPIE", "PDS_GRAPHIE", "PDS_TOTAL", "DOSE_EFF", "DOSE_PEAU",
        "TYPE_PROC", "VOIE_ACCES",
    ]
    n = min(len(cols_map), raw.shape[1])
    raw = raw.iloc[:, :n].copy()
    raw.columns = cols_map[:n]

    required = ["AGE", "IMC", "NB_IMAGES", "NB_SERIES",
                "PDS_TOTAL", "KERMA_mGy", "DOSE_EFF", "DOSE_PEAU",
                "DUREE_FLUO", "TYPE_PROC", "SEXE"]
    for c in required:
        if c not in raw.columns:
            return None

    # Nettoyage numérique
    float_cols = ["IMC", "PDS_TOTAL", "PDS_SCOPIE", "PDS_GRAPHIE",
                  "KERMA_mGy", "KERMA_Gy", "DOSE_EFF", "DOSE_PEAU",
                  "K_SCOPIE", "K_GRAPHIE"]
    int_cols   = ["AGE", "POIDS", "NB_IMAGES", "NB_SERIES", "FOV", "DIST_SOURCE"]
    for c in float_cols:
        if c in raw.columns:
            raw[c] = (raw[c].astype(str).str.replace(",", ".", regex=False)
                             .str.strip().pipe(pd.to_numeric, errors="coerce"))
    for c in int_cols:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Durée → minutes
    def _parse_dur(s):
        try:
            parts = str(s).strip().split(":")
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
            elif len(parts) == 2:
                return int(parts[0]) + int(parts[1]) / 60
            return float(s)
        except Exception:
            return np.nan

    raw["DUREE_MIN"] = raw["DUREE_FLUO"].apply(_parse_dur)

    # Variables binaires
    raw["PROC_TYPE"] = (raw["TYPE_PROC"].astype(str).str.strip().str.upper()
                        .apply(lambda x: "ANGIO" if "ANGIO" in x else "CORO"))
    raw["PROC_BIN"]  = (raw["PROC_TYPE"] == "ANGIO").astype(int)
    raw["SEXE_BIN"]  = (raw["SEXE"].astype(str).str.strip().str.upper() == "HOMME").astype(int)

    # Imputation médiane pour éviter les NaN dans les modèles
    num_cols = raw.select_dtypes(include=[np.number]).columns
    raw[num_cols] = raw[num_cols].fillna(raw[num_cols].median())

    return raw


@st.cache_data(show_spinner=False)
def load_data_from_path(filepath: str) -> pd.DataFrame | None:
    """Tente de charger le CSV depuis un chemin disque."""
    for enc in ["utf-8", "latin1", "cp1252", "utf-8-sig"]:
        for sep in [",", ";", "\t"]:
            try:
                raw = pd.read_csv(filepath, header=None, skiprows=6,
                                  encoding=enc, sep=sep, on_bad_lines="skip")
                if raw.shape[1] >= 28:
                    result = _build_df_from_raw(raw)
                    if result is not None:
                        return result
            except Exception:
                continue
    return None


@st.cache_data(show_spinner=False)
def load_data_from_upload(file_bytes: bytes) -> pd.DataFrame | None:
    """Tente de charger le CSV depuis un fichier uploadé via Streamlit."""
    for enc in ["utf-8", "latin1", "cp1252", "utf-8-sig"]:
        for sep in [",", ";", "\t"]:
            try:
                raw = pd.read_csv(io.BytesIO(file_bytes), header=None, skiprows=6,
                                  encoding=enc, sep=sep, on_bad_lines="skip")
                if raw.shape[1] >= 28:
                    result = _build_df_from_raw(raw)
                    if result is not None:
                        return result
            except Exception:
                continue
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODÈLES (figés une seule fois sur les 160 cas réels)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_models(_df: pd.DataFrame):
    """
    Entraîne :
      - 4 RandomForestRegressor (PDS_TOTAL, KERMA_mGy, DOSE_EFF, DOSE_PEAU)
      - 1 IsolationForest (détection de patients atypiques)
    Les modèles sont figés sur les 160 cas réels et ne sont jamais ré-entraînés.
    """
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X = _df[RF_FEATURES].copy()
    models, metrics = {}, {}
    for t in TARGETS:
        y = _df[t].copy()
        rf = RandomForestRegressor(n_estimators=300, max_depth=12,
                                   random_state=42, n_jobs=-1)
        rf.fit(X, y)
        y_pred = rf.predict(X)
        models[t] = rf
        metrics[t] = {
            "R²":   round(r2_score(y, y_pred), 3),
            "MAE":  round(mean_absolute_error(y, y_pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y, y_pred)), 2),
        }

    # IsolationForest — figé sur les 160 cas réels
    X_iso = _df[CLUST_FEATURES].fillna(_df[CLUST_FEATURES].median())
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_iso)

    return models, metrics, iso


@st.cache_resource(show_spinner=False)
def run_clustering(_df: pd.DataFrame, n_clusters: int = 4):
    """
    K-Means (n=4) + StandardScaler + PCA 3D.
    Les modèles sont figés sur les 160 cas réels.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    X = _df[CLUST_FEATURES].copy().fillna(_df[CLUST_FEATURES].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Elbow + Silhouette (sur les données réelles, pour l'affichage) ---
    inertias, sil_scores = [], []
    for k in range(2, 9):
        km_ = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl_ = km_.fit_predict(X_scaled)
        inertias.append(km_.inertia_)
        sil_scores.append(round(silhouette_score(X_scaled, lbl_), 3))

    # Clustering final (n=4)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    raw_labels = km.fit_predict(X_scaled)

    # Tri des clusters par PDS croissant (0 = faible, 3 = critique)
    tmp = pd.DataFrame({"raw": raw_labels, "pds": _df["PDS_TOTAL"].values})
    order = tmp.groupby("raw")["pds"].mean().sort_values()
    remap = {old: new for new, old in enumerate(order.index)}

    # PCA 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    return scaler, km, pca, remap, X_pca, raw_labels, inertias, sil_scores, var_exp


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FONCTIONS MÉTIER
# ══════════════════════════════════════════════════════════════════════════════

def classify_patient(pds: float, kerma: float, proc_type: str) -> str:
    """Retourne la classe ICRP interne (A/B/C/D)."""
    nrd_pds   = 45 if proc_type == "CORO" else 85
    nrd_kerma = 500 if proc_type == "CORO" else 1000
    if   pds > nrd_pds * 2   or kerma > 3000: score = 4
    elif pds > nrd_pds * 1.5 or kerma > 2000: score = 3
    elif pds > nrd_pds        or kerma > nrd_kerma: score = 2
    else: score = 1
    return {1: "A", 2: "B", 3: "C", 4: "D"}[score]


def get_recommendations(pds, kerma, duree, imc, proc_type, nb_images):
    """Retourne une liste (priorité, message) de recommandations cliniques."""
    recs = []
    nrd_pds = 45 if proc_type == "CORO" else 85
    if imc > 30:
        recs.append(("🔴 URGENT",  "IMC > 30 — Appliquer le protocole patient obèse (kVp adapté, détecteur rapproché)"))
    if duree > 25:
        recs.append(("🟠 ÉLEVÉ",   f"Durée {duree:.1f} min > 25 min — Réduire la fluoroscopie intermittente"))
    if kerma > 3000:
        recs.append(("🔴 URGENT",  f"Kérma {kerma:.0f} mGy > 3000 mGy — Effet déterministe probable — Dermatologue J0"))
    elif kerma > 2000:
        recs.append(("🔴 URGENT",  f"Kérma {kerma:.0f} mGy > 2000 mGy — Suivi dermatologique obligatoire J10–J14"))
    if pds > nrd_pds:
        recs.append(("🟠 ÉLEVÉ",   f"PDS {pds:.1f} Gy·cm² dépasse NRD {nrd_pds} Gy·cm² — Audit dosimétrique recommandé"))
    if nb_images > 600:
        recs.append(("🟡 MOYEN",   f"{nb_images:.0f} images — Limiter acquisitions non nécessaires (cible < 400)"))
    if not recs:
        recs.append(("✅ OK",       "Tous les indicateurs sont dans les normes. Maintenir les bonnes pratiques."))
    return recs


def assign_cluster_anomaly(df_in: pd.DataFrame, scaler, km, iso, pca, remap: dict) -> pd.DataFrame:
    """
    Applique (sans ré-entraîner) les modèles figés à n'importe quel DataFrame
    pour obtenir CLUSTER, ANOMALIE, SCORE_ANOMALIE et coordonnées PC1/PC2/PC3.
    """
    d = df_in.copy()
    X = d[CLUST_FEATURES].fillna(d[CLUST_FEATURES].median())
    Xs = scaler.transform(X)
    raw = km.predict(Xs)
    d["CLUSTER"]  = pd.Series(raw, index=d.index).map(remap)
    d["ANOMALIE"] = iso.predict(X)          # -1 = atypique, 1 = normal
    # Score brut : plus négatif = plus atypique
    d["SCORE_ANOMALIE"] = iso.score_samples(X)
    Xpca = pca.transform(Xs)
    d["PC1"], d["PC2"], d["PC3"] = Xpca[:, 0], Xpca[:, 1], Xpca[:, 2]
    return d


def _safe_encode(s: str) -> str:
    """Remplace les caractères non-latin-1 (accents, symboles, emojis) pour ReportLab.

    ReportLab/Helvetica ne sait pas rendre les emojis (🔴🟠🟡✅ etc.) : sans ce
    nettoyage, ils s'affichent comme des carrés noirs dans le PDF. On les
    remplace donc par un équivalent textuel sobre, adapté à un document médical.
    """
    table = str.maketrans({
        "é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a", "â": "a",
        "ù": "u", "û": "u", "î": "i", "ï": "i", "ô": "o", "ç": "c",
        "É": "E", "È": "E", "Ê": "E", "À": "A", "Â": "A", "Î": "I",
        "Ô": "O", "Ç": "C", "Ù": "U", "Û": "U", "²": "2",
        "≥": ">=", "≤": "<=", "→": "->", "·": ".",
        "★": "*", "•": "*", "°": "deg",
        # Emojis de risque / statut utilisés dans CLINICAL_RISK et les recommandations
        "✅": "[OK]", "🟡": "[!]", "🟠": "[!!]", "🔴": "[!!!]",
        "⚠": "[!]", "❌": "[X]", "🏥": "", "📋": "",
        "📄": "", "👤": "", "📐": "", "🚦": "", "👥": "", "💡": "",
        "🌍": "", "📊": "",
    })
    out = str(s).translate(table)
    # Filet de sécurité : supprime spécifiquement les emojis résiduels (plages
    # Unicode des pictogrammes/symboles) non couverts par la table ci-dessus,
    # sans toucher aux caractères typographiques usuels (tirets, guillemets...).
    def _is_emoji(ch: str) -> bool:
        cp = ord(ch)
        return (
            0x1F000 <= cp <= 0x1FFFF or  # emojis (smileys, symboles, transport...)
            0x2600  <= cp <= 0x27BF  or  # symboles divers / dingbats
            0x2190  <= cp <= 0x21FF  or  # flèches (hors celles déjà traduites)
            cp == 0xFE0F                # variation selector (emoji presentation)
        )
    return "".join(ch for ch in out if not _is_emoji(ch))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — GÉNÉRATION DU RAPPORT PDF PROFESSIONNEL
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(patient_data: dict, pds: float, kerma: float,
                        dose_eff: float, dose_peau: float, classe: str,
                        recs: list, cluster_num: int, cluster_name: str,
                        pct_rank: float, main_dataset: pd.DataFrame) -> bytes | None:
    """
    Génère un rapport PDF complet, professionnel et hiérarchisé.
    Retourne les bytes du PDF, ou None si ReportLab n'est pas installé.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable, Image as RLImage)
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    except ImportError:
        return None

    # ── Palette de couleurs — tons doux, professionnels ─────────────────────
    C_NAVY     = rl_colors.HexColor("#0f2942")   # bleu nuit institutionnel
    C_BLUE     = rl_colors.HexColor("#1d4e89")   # bleu principal (titres)
    C_BLUE_SFT = rl_colors.HexColor("#eaf1f8")   # bleu très pâle (fonds de bandeau)
    C_CYAN     = rl_colors.HexColor("#0ea5b5")
    C_TEXT     = rl_colors.HexColor("#22303f")
    C_MUTED    = rl_colors.HexColor("#6b7888")
    C_GREEN    = rl_colors.HexColor("#1f7a4d")
    C_GREEN_BG = rl_colors.HexColor("#e7f5ec")
    C_YELLOW   = rl_colors.HexColor("#92660a")
    C_YELLOW_BG= rl_colors.HexColor("#fdf3df")
    C_ORANGE   = rl_colors.HexColor("#b1530f")
    C_ORANGE_BG= rl_colors.HexColor("#fdebdc")
    C_RED_D    = rl_colors.HexColor("#a32626")
    C_RED_BG   = rl_colors.HexColor("#fbe7e7")
    C_WHITE    = rl_colors.white
    C_LIGHT    = rl_colors.HexColor("#f7f9fb")
    C_LIGHT2   = rl_colors.HexColor("#dbe3ec")
    C_LINE     = rl_colors.HexColor("#c7d2de")

    risk          = CLINICAL_RISK[classe]
    nrd_pds_ref   = 45 if patient_data["proc_type"] == "CORO" else 85
    nrd_kerma_ref = 500 if patient_data["proc_type"] == "CORO" else 1000
    pct_nrd       = pds / nrd_pds_ref * 100
    risk_palette  = {
        "A": (C_GREEN,  C_GREEN_BG),
        "B": (C_YELLOW, C_YELLOW_BG),
        "C": (C_ORANGE, C_ORANGE_BG),
        "D": (C_RED_D,  C_RED_BG),
    }
    risk_color, risk_bg = risk_palette.get(classe, (C_MUTED, C_LIGHT))
    now_str       = datetime.now().strftime("%d/%m/%Y à %H:%M")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=1.7*cm, leftMargin=1.7*cm,
                            topMargin=1.5*cm, bottomMargin=1.6*cm)

    styles = getSampleStyleSheet()
    S = lambda name, **kw: ParagraphStyle(name, parent=styles["Normal"], **kw)

    s_inst      = S("INST", fontSize=8.5, textColor=C_MUTED, fontName="Helvetica",
                    leading=11)
    s_inst_bold = S("INSTB", fontSize=12.5, textColor=C_NAVY, fontName="Helvetica-Bold",
                    leading=15)
    s_doctitle  = S("DT", fontSize=15, textColor=C_BLUE, fontName="Helvetica-Bold",
                    alignment=TA_RIGHT, leading=18)
    s_docsub    = S("DS", fontSize=9, textColor=C_MUTED, fontName="Helvetica",
                    alignment=TA_RIGHT, leading=12)
    s_h2        = S("H2", fontSize=11.5, textColor=C_WHITE, fontName="Helvetica-Bold",
                    leading=14)
    s_h3        = S("H3", fontSize=9.5, textColor=C_BLUE, fontName="Helvetica-Bold",
                    spaceBefore=6, spaceAfter=3, leading=12)
    s_body      = S("B",  fontSize=9.2, textColor=C_TEXT, spaceAfter=3, leading=13.5)
    s_body_ctr  = S("BC", fontSize=9.2, textColor=C_TEXT, spaceAfter=3, leading=13.5,
                    alignment=TA_CENTER)
    s_small     = S("SM", fontSize=8, textColor=C_MUTED, spaceAfter=2, leading=11)
    s_risk_lbl  = S("RL", fontSize=15, textColor=risk_color, fontName="Helvetica-Bold",
                    alignment=TA_CENTER, spaceAfter=2, leading=18)
    s_risk_lbl_sm = S("RLS", fontSize=11, textColor=risk_color, fontName="Helvetica-Bold",
                    alignment=TA_CENTER, spaceAfter=2, leading=13.5)
    s_risk_desc = S("RD", fontSize=9, textColor=C_TEXT, alignment=TA_CENTER,
                    spaceAfter=2, leading=12.5)
    s_warn      = S("W",  fontSize=9, textColor=C_ORANGE, spaceAfter=2, leading=12.5)
    s_legal     = S("LG", fontSize=7.8, textColor=C_MUTED, alignment=TA_CENTER,
                    leading=11)

    def section_header(num_label: str, title_txt: str):
        """Bandeau de section bleu, pleine largeur, avec numéro et titre en blanc."""
        cell = Paragraph(_safe_encode(f"{num_label}   {title_txt}"), s_h2)
        t = Table([[cell]], colWidths=[17.1*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_BLUE),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return t

    def tbl_style(header_color=C_NAVY, zebra=True):
        style = [
            ("BACKGROUND",    (0, 0), (-1,  0), header_color),
            ("TEXTCOLOR",     (0, 0), (-1,  0), C_WHITE),
            ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8.3),
            ("GRID",          (0, 0), (-1, -1), 0.4, C_LINE),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ]
        if zebra:
            style.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]))
        return TableStyle(style)

    def risk_pill(label_text: str, dot_color, text_style, dot_size=9):
        """
        Construit un badge visuel : pastille pleine colorée + texte, alignés
        horizontalement et correctement centrés. Remplace les emojis (non
        supportés par ReportLab) par un vrai indicateur graphique vectoriel.
        """
        from reportlab.graphics.shapes import Drawing, Circle
        from reportlab.pdfbase.pdfmetrics import stringWidth

        clean_text = _safe_encode(label_text)
        d = Drawing(dot_size + 4, dot_size + 2)
        d.add(Circle((dot_size/2) + 2, (dot_size/2) + 1, dot_size/2,
                     fillColor=dot_color, strokeColor=None))
        txt = Paragraph(clean_text, text_style)

        dot_col_w = dot_size + 6
        text_w = stringWidth(clean_text, text_style.fontName, text_style.fontSize) + 6

        t = Table([[d, txt]], colWidths=[dot_col_w, text_w])
        t.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",         (0, 0), (0, 0), "CENTER"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        t.hAlign = "CENTER"
        return t

    def info_box(text_html: str, bg=C_BLUE_SFT, border=C_BLUE, text_style=None):
        """Encadré coloré avec liseré gauche, pour mettre en valeur un message."""
        st_use = text_style or s_body
        p = Paragraph(_safe_encode(text_html), st_use)
        t = Table([[p]], colWidths=[17.1*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, -1), bg),
            ("LINEBEFORE",     (0, 0), (0, -1), 3, border),
            ("LEFTPADDING",    (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 10),
            ("TOPPADDING",     (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 8),
        ]))
        return t

    story = []

    # ══════════════════════════════════════════════════════════════════════
    # EN-TÊTE — LOGO CHU (gauche) + IDENTITÉ INSTITUTIONNELLE + TITRE (droite)
    # ══════════════════════════════════════════════════════════════════════
    try:
        _logo_buf = io.BytesIO(get_logo_bytes())
        logo_img  = RLImage(_logo_buf, width=1.7*cm, height=1.7*cm)
    except Exception:
        logo_img = Paragraph("", s_small)

    inst_block = [
        Paragraph(_safe_encode("Centre Hospitalo-Universitaire"), s_inst),
        Paragraph(_safe_encode("Mustapha Bacha — Alger"), s_inst_bold),
        Paragraph(_safe_encode("Service de Cardiologie Interventionnelle"), s_inst),
    ]
    title_block = [
        Paragraph(_safe_encode("Rapport de Radioprotection"), s_doctitle),
        Paragraph(_safe_encode("CardioDose Optimizer "), s_docsub),
        Paragraph(_safe_encode(f"Généré le {now_str}"), s_docsub),
    ]

    header_tbl = Table([[logo_img, inst_block, title_block]],
                       colWidths=[2.1*cm, 8*cm, 7*cm])
    header_tbl.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1.6, color=C_BLUE))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_CYAN, spaceBefore=1))
    story.append(Spacer(1, 0.45*cm))

    # ══════════════════════════════════════════════════════════════════════
    # BANDEAU RÉSUMÉ — Patient, procédure, classe de risque (vue d'ensemble)
    # ══════════════════════════════════════════════════════════════════════
    summary_rows = [[
        Paragraph(_safe_encode("PATIENT"), s_small),
        Paragraph(_safe_encode("PROCÉDURE"), s_small),
        Paragraph(_safe_encode("ÉVALUATION DU RISQUE"), s_small),
    ], [
        Paragraph(_safe_encode(
            f"{patient_data.get('prenom','')} {patient_data.get('nom','')}".strip() or
            f"{patient_data.get('sexe','-')}, {patient_data.get('age','-')} ans" +
            (f" — IPP : {patient_data.get('ipp')}" if patient_data.get('ipp') else "") +
            f"\n{patient_data.get('sexe','-')}, {patient_data.get('age','-')} ans — IMC {patient_data.get('imc',0):.1f}"
        ), s_body_ctr),
        Paragraph(_safe_encode(f"{patient_data.get('proc_type','-')} — "
                                f"{patient_data.get('duree','-')} min de scopie"), s_body_ctr),
        risk_pill(f"{risk['label']} (Classe {classe})", risk_color, s_risk_lbl),
    ]]
    summary_tbl = Table(summary_rows, colWidths=[5.7*cm, 5.7*cm, 5.7*cm])
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, -1), C_LIGHT),
        ("BOX",            (0, 0), (-1, -1), 0.6, C_LINE),
        ("INNERGRID",      (0, 0), (-1, -1), 0.4, C_LINE),
        ("BACKGROUND",     (2, 1), (2, 1), risk_bg),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",     (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING",  (0, 0), (-1, 0), 2),
        ("TOPPADDING",     (0, 1), (-1, 1), 4),
        ("BOTTOMPADDING",  (0, 1), (-1, 1), 10),
    ]))
    story.append(summary_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════════════
    # 1. PROFIL CLINIQUE DU PATIENT
    # ══════════════════════════════════════════════════════════════════════
    story.append(section_header("1", "Profil Clinique du Patient"))
    story.append(Spacer(1, 0.25*cm))

    comorb = []
    if patient_data.get("diabete"): comorb.append("Diabète")
    if patient_data.get("hta"):     comorb.append("HTA")
    if patient_data.get("fumeur"):  comorb.append("Fumeur")
    if patient_data.get("cardio"):  comorb.append("Cardiopathie")
    comorb_str = ", ".join(comorb) if comorb else "Aucune comorbidité déclarée"

    pt_rows = [
        [_safe_encode("Paramètre"), _safe_encode("Valeur"),
         _safe_encode("Paramètre"), _safe_encode("Valeur")],
        [_safe_encode("Nom"),        _safe_encode(patient_data.get("nom", "-") or "-"),
         _safe_encode("IPP / N° dossier"), _safe_encode(patient_data.get("ipp", "-") or "-")],
        [_safe_encode("Prénom"),     _safe_encode(patient_data.get("prenom", "-") or "-"),
         _safe_encode("Type de procédure"), patient_data.get("proc_type", "-")],
        [_safe_encode("Sexe"),       patient_data.get("sexe", "-"),
         _safe_encode("Durée de scopie"), f"{patient_data.get('duree', '-')} min"],
        [_safe_encode("Âge"),        f"{patient_data.get('age', '-')} ans",
         _safe_encode("Nombre de séries"),  str(patient_data.get("nb_series", "-"))],
        [_safe_encode("Poids"),      f"{patient_data.get('poids', '-')} kg",
         _safe_encode("Nombre d'images"),  str(patient_data.get("nb_images", "-"))],
        [_safe_encode("Taille"),     f"{patient_data.get('taille', '-')} m",
         _safe_encode("IMC"), f"{patient_data.get('imc', 0):.1f} kg/m2"],
        [_safe_encode("Comorbidités"), _safe_encode(comorb_str),
         "", ""],
    ]
    t_pt = Table(pt_rows, colWidths=[3.8*cm, 4.7*cm, 4.2*cm, 4.4*cm])
    t_pt.setStyle(tbl_style())
    story.append(t_pt)
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════════════
    # 2. RÉSULTATS DOSIMÉTRIQUES — Prédiction Random Forest
    # ══════════════════════════════════════════════════════════════════════
    story.append(section_header("2", "Résultats Dosimétriques (Prédiction IA)"))
    story.append(Spacer(1, 0.25*cm))

    pred_rows = [
        [_safe_encode("Indicateur"), _safe_encode("Valeur Prédite"),
         _safe_encode("NRD Référence"), _safe_encode("% du NRD"), _safe_encode("Statut")],
        [_safe_encode("PDS Total (Gy.cm²)"), f"{pds:.1f}",
         f"{nrd_pds_ref} Gy.cm2", f"{pct_nrd:.0f}%",
         _safe_encode("Conforme" if pds <= nrd_pds_ref else "Dépassé")],
        [_safe_encode("Kérma Air (mGy)"), f"{kerma:.0f}",
         f"{nrd_kerma_ref} mGy", f"{kerma/nrd_kerma_ref*100:.0f}%",
         _safe_encode("OK" if kerma <= nrd_kerma_ref else "Élevé")],
        [_safe_encode("Dose Efficace (mSv)"), f"{dose_eff:.2f}", "-", "-", "-"],
        [_safe_encode("Dose Peau (Gy)"),      f"{dose_peau:.2f}",
         "2.0 Gy (seuil érythème)", "-",
         _safe_encode("Suivi requis" if dose_peau >= 2 else "OK")],
    ]
    t_pred = Table(pred_rows, colWidths=[4.4*cm, 3*cm, 3.6*cm, 2.3*cm, 3.8*cm])
    t_pred_style = tbl_style()
    # Mise en valeur conditionnelle du statut (vert / rouge)
    if pds <= nrd_pds_ref:
        t_pred_style.add("BACKGROUND", (4, 1), (4, 1), C_GREEN_BG)
        t_pred_style.add("TEXTCOLOR",  (4, 1), (4, 1), C_GREEN)
    else:
        t_pred_style.add("BACKGROUND", (4, 1), (4, 1), C_RED_BG)
        t_pred_style.add("TEXTCOLOR",  (4, 1), (4, 1), C_RED_D)
    if kerma <= nrd_kerma_ref:
        t_pred_style.add("BACKGROUND", (4, 2), (4, 2), C_GREEN_BG)
        t_pred_style.add("TEXTCOLOR",  (4, 2), (4, 2), C_GREEN)
    else:
        t_pred_style.add("BACKGROUND", (4, 2), (4, 2), C_ORANGE_BG)
        t_pred_style.add("TEXTCOLOR",  (4, 2), (4, 2), C_ORANGE)
    if dose_peau >= 2:
        t_pred_style.add("BACKGROUND", (4, 4), (4, 4), C_RED_BG)
        t_pred_style.add("TEXTCOLOR",  (4, 4), (4, 4), C_RED_D)
    else:
        t_pred_style.add("BACKGROUND", (4, 4), (4, 4), C_GREEN_BG)
        t_pred_style.add("TEXTCOLOR",  (4, 4), (4, 4), C_GREEN)
    t_pred.setStyle(t_pred_style)
    story.append(t_pred)
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════════════
    # 3. ÉVALUATION DU RISQUE & POSITION DANS LA COHORTE
    # ══════════════════════════════════════════════════════════════════════
    story.append(section_header("3", "Évaluation du Risque & Position dans la Cohorte"))
    story.append(Spacer(1, 0.25*cm))

    risk_block = [
        risk_pill(f"{risk['label']}  —  Classe ICRP {classe}", risk_color, s_risk_lbl),
        Paragraph(_safe_encode(risk["desc"]), s_risk_desc),
    ]
    risk_tbl = Table([[risk_block]], colWidths=[17.1*cm])
    risk_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), risk_bg),
        ("BOX",           (0, 0), (-1, -1), 0.7, risk_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(risk_tbl)
    story.append(Spacer(1, 0.3*cm))
    story.append(info_box(
        f"Percentile cohorte : <b>{pct_rank:.0f}<sup>e</sup></b> — Ce patient est plus irradié "
        f"que {pct_rank:.0f}% des patients de la cohorte de référence "
        f"({len(main_dataset)} patients, réels et simulés).",
        bg=C_LIGHT, border=C_MUTED,
    ))
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════════════
    # 4. PROFIL CLINIQUE PAR SIMILARITÉ (CLUSTERING)
    # ══════════════════════════════════════════════════════════════════════
    story.append(section_header("4", "Classification par Similarité (Clustering)"))
    story.append(Spacer(1, 0.25*cm))

    CLUSTER_CLINICAL_DESC_PDF = {
        0: ("Ce patient présente une exposition aux rayonnements inférieure à la moyenne du "
            "service. La durée de la procédure, le nombre d'images et la dose reçue sont tous "
            "modérés. Aucune action particulière n'est requise au-delà des bonnes pratiques "
            "habituelles."),
        1: ("Ce patient présente une exposition modérée, comparable à la majorité des patients "
            "de la cohorte. Une surveillance standard est recommandée. Vérifier que la durée de "
            "scopie reste dans les limites habituelles."),
        2: ("Ce patient présente une exposition supérieure à la moyenne. La procédure a "
            "nécessité plus d'acquisitions ou une durée plus longue que la majorité. Un audit "
            "dosimétrique est recommandé et une surveillance cutanée à J10-J14 est conseillée."),
        3: ("Ce patient présente une exposition critique, parmi les plus élevées de la cohorte. "
            "La dose reçue dépasse significativement les niveaux de référence. Un suivi "
            "dermatologique immédiat est obligatoire et un audit de la procédure est requis."),
    }
    CLUSTER_RECO_PDF = {
        0: "Maintenir les bonnes pratiques. Aucune action dosimétrique spécifique.",
        1: "Surveillance standard. Vérifier la durée de scopie à chaque procédure.",
        2: "Audit dosimétrique recommandé. Surveillance cutanée J10-J14 si dose peau > 1 Gy.",
        3: "Suivi dermatologique obligatoire. Révision urgente du protocole de la procédure.",
    }

    story.append(Paragraph(
        _safe_encode(f"Profil identifié : Cluster {cluster_num} — {cluster_name}"), s_h3))
    story.append(Paragraph(
        _safe_encode(CLUSTER_CLINICAL_DESC_PDF.get(cluster_num,
            "Profil dosimétrique identifié parmi les 4 groupes de la cohorte.")), s_body))
    story.append(Spacer(1, 0.15*cm))
    story.append(info_box(
        f"<b>Recommandation associée au profil :</b> {CLUSTER_RECO_PDF.get(cluster_num, '-')}",
        bg=C_YELLOW_BG, border=C_YELLOW, text_style=s_warn,
    ))
    story.append(Spacer(1, 0.3*cm))

    cluster_rows = [
        [_safe_encode("Profil"), _safe_encode("Nom clinique"),
         _safe_encode("PDS moy (Gy.cm2)"), _safe_encode("Kerma moy (mGy)"),
         _safe_encode("Durée moy (min)")],
    ]
    for ci, cname in CLUSTER_NAMES.items():
        marker = " (CE PATIENT)" if ci == cluster_num else ""
        cluster_rows.append([
            str(ci), _safe_encode(cname + marker),
            "-", "-", "-",
        ])
    t_cl = Table(cluster_rows, colWidths=[1.5*cm, 6.8*cm, 3*cm, 3*cm, 2.8*cm])
    t_cl_style = tbl_style()
    t_cl_style.add("BACKGROUND", (0, cluster_num + 1), (-1, cluster_num + 1), C_BLUE_SFT)
    t_cl_style.add("FONTNAME", (0, cluster_num + 1), (-1, cluster_num + 1), "Helvetica-Bold")
    t_cl.setStyle(t_cl_style)
    story.append(t_cl)
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════════════
    # 5. RECOMMANDATIONS PRIORITAIRES
    # ══════════════════════════════════════════════════════════════════════
    story.append(section_header("5", "Recommandations Prioritaires"))
    story.append(Spacer(1, 0.25*cm))

    if recs:
        rec_rows = [[_safe_encode("Priorité"), _safe_encode("Recommandation clinique")]]
        prio_bg_map = {
            "URGENT": C_RED_BG, "ÉLEVÉ": C_ORANGE_BG, "ELEVE": C_ORANGE_BG,
            "MOYEN": C_YELLOW_BG,
        }
        for prio, msg in recs:
            rec_rows.append([_safe_encode(prio.replace("🔴", "").replace("🟠", "")
                                          .replace("🟡", "").strip()),
                             _safe_encode(msg)])
        t_rec = Table(rec_rows, colWidths=[3.3*cm, 13.8*cm])
        t_rec_style = tbl_style()
        for i, (prio, _msg) in enumerate(recs, start=1):
            for key, bg in prio_bg_map.items():
                if key in prio:
                    t_rec_style.add("BACKGROUND", (0, i), (0, i), bg)
                    break
        t_rec.setStyle(t_rec_style)
        story.append(t_rec)
    else:
        story.append(info_box(
            "Aucune recommandation spécifique — les paramètres de cette procédure restent "
            "dans les marges attendues.", bg=C_GREEN_BG, border=C_GREEN,
        ))
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════════════
    # 6. CONFORMITÉ AUX NRD INTERNATIONAUX
    # ══════════════════════════════════════════════════════════════════════
    story.append(section_header("6", "Conformité aux NRD Internationaux"))
    story.append(Spacer(1, 0.25*cm))

    nrd_rows = [[
        _safe_encode("Organisme"), _safe_encode("NRD PDS"),
        _safe_encode("PDS Patient"), _safe_encode("% NRD"),
        _safe_encode("NRD Kérma"), _safe_encode("Kérma Patient"), _safe_encode("Statut"),
    ]]
    nrd_status_idx = []
    for org, vals in NRD.items():
        nrd_pds_v   = vals[f"PDS_{patient_data.get('proc_type', 'CORO')}"]
        nrd_kerma_v = vals[f"Kerma_{patient_data.get('proc_type', 'CORO')}"]
        conforme = pds <= nrd_pds_v
        statut = _safe_encode("Conforme" if conforme else "Dépassé")
        nrd_status_idx.append(conforme)
        nrd_rows.append([
            _safe_encode(org), f"{nrd_pds_v} Gy.cm2",
            f"{pds:.1f}", f"{pds/nrd_pds_v*100:.0f}%",
            f"{nrd_kerma_v} mGy", f"{kerma:.0f}", statut,
        ])
    t_nrd = Table(nrd_rows, colWidths=[3*cm, 2.4*cm, 2.1*cm, 1.7*cm, 2.4*cm, 2.3*cm, 2.2*cm])
    t_nrd_style = tbl_style()
    for i, ok in enumerate(nrd_status_idx, start=1):
        if ok:
            t_nrd_style.add("BACKGROUND", (6, i), (6, i), C_GREEN_BG)
            t_nrd_style.add("TEXTCOLOR",  (6, i), (6, i), C_GREEN)
        else:
            t_nrd_style.add("BACKGROUND", (6, i), (6, i), C_RED_BG)
            t_nrd_style.add("TEXTCOLOR",  (6, i), (6, i), C_RED_D)
    t_nrd.setStyle(t_nrd_style)
    story.append(t_nrd)
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════════════
    # 7. CONTEXTE — STATISTIQUES DE LA COHORTE DU SERVICE
    # ══════════════════════════════════════════════════════════════════════
    story.append(section_header("7", "Contexte — Statistiques de la Cohorte du Service"))
    story.append(Spacer(1, 0.25*cm))

    n_real = (main_dataset["SOURCE"] == "Réel").sum()
    n_sim  = (main_dataset["SOURCE"] == "Simulé").sum()
    nrd_ref_map = {"CORO": 45, "ANGIO": 85}
    pct_dep = main_dataset.apply(
        lambda r: r["PDS_TOTAL"] > nrd_ref_map.get(r["PROC_TYPE"], 45), axis=1).mean() * 100

    stat_rows = [
        [_safe_encode("Indicateur"), _safe_encode("Valeur")],
        [_safe_encode("Total patients (réels + simulés)"), f"{len(main_dataset)} ({n_real} réels, {n_sim} simulés)"],
        [_safe_encode("PDS médian de la cohorte"),  f"{main_dataset['PDS_TOTAL'].median():.1f} Gy.cm2"],
        [_safe_encode("Kérma médian de la cohorte"), f"{main_dataset['KERMA_mGy'].median():.0f} mGy"],
        [_safe_encode("Dose efficace médiane"), f"{main_dataset['DOSE_EFF'].median():.2f} mSv"],
        [_safe_encode("% patients > NRD (France ASN)"), f"{pct_dep:.1f}%"],
        [_safe_encode("PDS de ce patient"),  f"{pds:.1f} Gy.cm2"],
        [_safe_encode("Percentile PDS de ce patient"),     f"{pct_rank:.0f}e percentile"],
    ]
    t_stat = Table(stat_rows, colWidths=[10.5*cm, 6.6*cm])
    t_stat.setStyle(tbl_style())
    story.append(t_stat)
    story.append(Spacer(1, 0.6*cm))

    # ══════════════════════════════════════════════════════════════════════
    # CONCLUSION & MENTION LÉGALE
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE))
    story.append(Spacer(1, 0.25*cm))
    if classe == "A":
        conclusion_suite = "Aucun suivi particulier n'est requis au-delà des bonnes pratiques habituelles."
    else:
        conclusion_suite = "Il est recommandé de suivre les actions prioritaires détaillées en section 5 ci-dessus."
    story.append(info_box(
        f"<b>Conclusion :</b> au regard des résultats dosimétriques et du profil clinique "
        f"observé, ce patient est classé en <b>{risk['label']}</b> (classe {classe}). "
        f"{conclusion_suite}",
        bg=C_BLUE_SFT, border=C_BLUE,
    ))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=0.6, color=C_LIGHT2))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        _safe_encode("CardioDose Optimizer — Centre Hospitalo Universitaire Mustapha, Alger — "
                     "Service de Cardiologie Interventionnelle"), s_legal))
    story.append(Paragraph(
        _safe_encode("Ce rapport est un outil d'aide à la décision dosimétrique. Il ne remplace "
                     "en aucun cas le jugement médical qualifié du cardiologue ou du radiologue "
                     "référent, seuls habilités à statuer sur la conduite à tenir."),
        s_legal))
    story.append(Paragraph(_safe_encode(f"Document généré automatiquement le {now_str} "
                                        f"— CardioDose Optimizer v6.0"), s_legal))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — INITIALISATION DE L'APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

# --- Chargement du CSV ---
CSV_CANDIDATES = [
    "Classeur_du_bino_me_GRISSI_et_GAYA.csv",
    "data.csv",
    "cardiodose.csv",
]

df_real = None

# 1) Tentative sur les chemins candidats locaux
for path in CSV_CANDIDATES:
    df_real = load_data_from_path(path)
    if df_real is not None:
        break

# 2) Si pas trouvé → widget d'upload Streamlit (ne bloque pas, propose l'upload)
if df_real is None:
    st.markdown("""
    <div style="background:#172554;border:1px solid #1d4ed8;border-radius:12px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="font-size:16px;font-weight:700;color:#60a5fa;margin-bottom:8px;">
            📂 Fichier CSV non détecté automatiquement
        </div>
        <div style="font-size:13px;color:#94a3b8;line-height:1.7;">
            Placez le fichier <b style="color:#e2e8f0;">Classeur_du_bino_me_GRISSI_et_GAYA.csv</b>
            dans le même dossier que <code>fasi.py</code>, ou uploadez-le ci-dessous.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📁 Uploader le fichier CSV du dataset",
        type=["csv"],
        help="Format attendu : CSV avec en-têtes sur la ligne 7 (skiprows=6)",
    )
    if uploaded_file is not None:
        with st.spinner("Chargement et validation du fichier…"):
            df_real = load_data_from_upload(uploaded_file.read())
        if df_real is None:
            st.error("❌ Impossible de parser ce fichier CSV. "
                     "Vérifiez le format (séparateur, encodage, lignes d'en-tête).")
            st.stop()
        else:
            st.success(f"✅ Dataset chargé : {len(df_real)} patients.")
    else:
        st.info("👆 En attente du fichier CSV. L'application démarrera dès l'upload.")
        st.stop()

# --- Entraînement des modèles (figés une seule fois) ---
with st.spinner("🤖 Entraînement des modèles IA sur les données réelles…"):
    models, rf_metrics, iso_model = train_models(df_real)
    (cl_scaler, km_model, pca_model, CLUSTER_REMAP,
     X_pca_real, raw_labels_real,
     elbow_inertias, sil_scores, pca_var_exp) = run_clustering(df_real)

# --- Classification ICRP initiale (sur les 160 cas réels) ---
df_real = df_real.copy()
df_real["CLASSE"] = df_real.apply(
    lambda r: classify_patient(r["PDS_TOTAL"], r["KERMA_mGy"], r["PROC_TYPE"]), axis=1
)

# --- Dataset principal (réel + futurs simulés) ---
df_real_main = df_real.copy()
df_real_main["SOURCE"] = "Réel"
df_real_main = df_real_main[MAIN_COLS]
df_real_main["N"] = df_real_main["N"].astype(str)

# --- Session state ---
_defaults = {
    "main_dataset":         df_real_main.copy(),
    "predictions_history":  [],
    "pred_pds":             None,
    "pred_kerma":           None,
    "pred_dose_eff":        None,
    "pred_dose_peau":       None,
    "classe":               None,
    "patient_data":         None,
    "show_new_pred_hint":   False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

n_sim_total = (st.session_state.main_dataset["SOURCE"] == "Simulé").sum()
_logo_uri = get_logo_data_uri()
st.markdown(f"""
<div class="main-header">
  <img src="{_logo_uri}" style="width:52px;height:52px;border-radius:10px;
       background:#ffffff;padding:4px;flex-shrink:0;" alt="Logo CHU Mustapha Bacha" />
  <div>
    <div style="font-size:22px;font-weight:800;color:#22d3ee;letter-spacing:-0.02em">
      CardioDose Optimizer <span style="color:#3b82f6"> </span>
    </div>
    <div style="font-size:12px;color:#475569;margin-top:2px">
      CHU Mustapha Bacha, Alger — Aide à la décision dosimétrique — Cardiologie Interventionnelle |
      <b style="color:#94a3b8">{len(df_real)}</b> procédures réelles ·
      <b style="color:#a78bfa">{n_sim_total}</b> simulées cette session
    </div>
  </div>
  <div style="margin-left:auto;text-align:right">
    <div style="font-size:10px;color:#334155">Dernière mise à jour</div>
    <div style="font-size:12px;color:#22d3ee;font-weight:600">{datetime.now().strftime("%d/%m/%Y %H:%M")}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SIDEBAR (paramètres patient — persistant sur tous les onglets)
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;padding:6px 4px 16px;
                border-bottom:1px solid #1e3a5f;margin-bottom:16px;">
      <img src="{_logo_uri}" style="width:42px;height:42px;border-radius:8px;
           background:#ffffff;padding:3px;flex-shrink:0;" alt="Logo CHU" />
      <div>
        <div style="font-size:12px;font-weight:700;color:#cbd5e1;line-height:1.3">
          Centre Hospitalo Universitaire Mustapaha 
        </div>
        <div style="font-size:10px;color:#64748b;line-height:1.3">
          Alger — Cardiologie Interventionnelle
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🪪 Identité du Patient</div>', unsafe_allow_html=True)
    patient_nom    = st.text_input("Nom", placeholder="NOM")
    patient_prenom = st.text_input("Prénom", placeholder="Prénom")
    patient_ipp    = st.text_input("IPP / N° dossier", placeholder="ex: 2024-00123")

    st.markdown('<div class="section-title">🧑‍⚕️ Paramètres Patient</div>', unsafe_allow_html=True)

    sexe  = st.selectbox("Sexe", ["FEMME", "HOMME"])
    age   = st.slider("Âge (ans)", 18, 90, 58)
    poids = st.slider("Poids (kg)", 40, 150, 75)
    taille = st.slider("Taille (m)", 1.40, 2.00, 1.65, 0.01)
    imc   = round(poids / taille ** 2, 1)
    st.metric("IMC calculé", f"{imc} kg/m²",
              delta="Obèse" if imc >= 30 else ("Surpoids" if imc >= 25 else "Normal"),
              delta_color="inverse" if imc >= 30 else "off")

    st.markdown('<div class="section-title">🩺 Procédure</div>', unsafe_allow_html=True)
    proc_type = st.selectbox("Type", ["CORO", "ANGIO"])
    duree     = st.slider("Durée scopie (min)", 1, 90, 12)
    nb_series = st.slider("Nb séries", 1, 60, 6)
    nb_images = st.slider("Nb images", 50, 3000, 350)

    st.markdown('<div class="section-title">🩻 Comorbidités</div>', unsafe_allow_html=True)
    diabete = st.checkbox("Diabète")
    hta     = st.checkbox("HTA")
    fumeur  = st.checkbox("Fumeur")
    cardio  = st.checkbox("Cardiopathie")
    st.caption("_Variables enregistrées pour le suivi clinique (p>0.05 sur la dose)._")

    st.markdown("---")
    sidebar_predict = st.button("🔮 Prédire la dose", type="primary",
                                use_container_width=True, key="predict_sidebar")
    st.caption(f"💾 {len(st.session_state.predictions_history)} simulation(s) enregistrée(s).")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ONGLETS
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "🏠 Accueil & Prédiction",
    "📄 Rapport PDF",
    "📊 Mes Simulations",
    "📈 Données du Service",
    "⚠️ Atypiques & Risques",
    "🌍 Conformité NRD",
    "📊 Analyse des Profils Cliniques",
])

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 0 — ACCUEIL & PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:

    st.markdown('<div class="cta-subtitle">Renseignez les paramètres dans la barre latérale puis '
                'lancez la prédiction</div>', unsafe_allow_html=True)

    predict_main = st.button("🔮  PRÉDIRE LA DOSE DE CE PATIENT",
                             type="primary", use_container_width=True, key="predict_main")

    # ── Calcul ────────────────────────────────────────────────────────────────
    if predict_main or sidebar_predict:
        st.session_state.show_new_pred_hint = False
        with st.spinner("🤖 Calcul Random Forest en cours…"):
            sexe_bin = 1 if sexe == "HOMME" else 0
            proc_bin = 1 if proc_type == "ANGIO" else 0
            X_new = pd.DataFrame(
                [[age, sexe_bin, imc, duree, nb_series, nb_images, proc_bin]],
                columns=RF_FEATURES,
            )
            preds = {t: float(models[t].predict(X_new)[0]) for t in models}

        pds_v   = round(preds["PDS_TOTAL"], 2)
        kerma_v = round(preds["KERMA_mGy"], 1)
        eff_v   = round(preds["DOSE_EFF"], 2)
        peau_v  = round(preds["DOSE_PEAU"], 2)
        classe_v = classify_patient(pds_v, kerma_v, proc_type)

        # Cluster du nouveau patient (sans ré-entraîner)
        X_new_cl = pd.DataFrame(
            [[imc, duree, nb_series, nb_images, pds_v, kerma_v]],
            columns=CLUST_FEATURES,
        )
        Xs_new = cl_scaler.transform(X_new_cl)
        cluster_raw = km_model.predict(Xs_new)[0]
        cluster_num = CLUSTER_REMAP.get(cluster_raw, 0)

        # Détection atypique
        iso_score   = iso_model.predict(X_new_cl)[0]   # -1 = atypique
        is_atypique = iso_score == -1

        st.session_state.pred_pds       = pds_v
        st.session_state.pred_kerma     = kerma_v
        st.session_state.pred_dose_eff  = eff_v
        st.session_state.pred_dose_peau = peau_v
        st.session_state.classe         = classe_v
        st.session_state.cluster_num    = cluster_num
        st.session_state.is_atypique    = is_atypique
        st.session_state.patient_data   = {
            "nom": patient_nom, "prenom": patient_prenom, "ipp": patient_ipp,
            "age": age, "sexe": sexe, "imc": imc, "poids": poids, "taille": taille,
            "proc_type": proc_type, "duree": duree, "nb_series": nb_series,
            "nb_images": nb_images, "diabete": diabete, "hta": hta,
            "fumeur": fumeur, "cardio": cardio,
        }

        # Ajout dynamique au dataset principal
        n_sim = len(st.session_state.predictions_history) + 1
        new_row = {
            "N": f"S{n_sim}", "SOURCE": "Simulé", "SEXE": sexe, "AGE": age,
            "IMC": imc, "PROC_TYPE": proc_type, "DUREE_MIN": duree,
            "NB_SERIES": nb_series, "NB_IMAGES": nb_images,
            "PDS_TOTAL": pds_v, "KERMA_mGy": kerma_v,
            "DOSE_EFF": eff_v, "DOSE_PEAU": peau_v, "CLASSE": classe_v,
        }
        st.session_state.main_dataset = pd.concat(
            [st.session_state.main_dataset, pd.DataFrame([new_row])],
            ignore_index=True,
        )
        st.session_state.predictions_history.append({
            "Heure": datetime.now().strftime("%H:%M:%S"),
            "Sexe": sexe, "Âge": age, "IMC": imc, "Procédure": proc_type,
            "Durée (min)": duree, "Nb séries": nb_series, "Nb images": nb_images,
            "PDS (Gy·cm²)": pds_v, "Kérma (mGy)": kerma_v,
            "Dose Eff (mSv)": eff_v, "Dose Peau (Gy)": peau_v,
            "Classe": classe_v, "Risque": CLINICAL_RISK[classe_v]["label"],
            "Cluster": cluster_num, "Atypique": "Oui" if is_atypique else "Non",
            "Diabète": "Oui" if diabete else "Non", "HTA": "Oui" if hta else "Non",
            "Fumeur": "Oui" if fumeur else "Non", "Cardiopathie": "Oui" if cardio else "Non",
        })

    # ── Affichage des résultats ───────────────────────────────────────────────
    if st.session_state.pred_pds is None:
        st.markdown("""
        <div class="alert-info">
            ℹ️ <b>Aucune prédiction encore effectuée.</b>
            Renseignez les paramètres dans la barre latérale et cliquez sur le bouton ci-dessus.
        </div>
        """, unsafe_allow_html=True)
    else:
        pds       = st.session_state.pred_pds
        kerma     = st.session_state.pred_kerma
        dose_eff  = st.session_state.pred_dose_eff
        dose_peau = st.session_state.pred_dose_peau
        classe    = st.session_state.classe
        risk      = CLINICAL_RISK[classe]
        pd_data   = st.session_state.patient_data
        cluster_num  = st.session_state.get("cluster_num", 0)
        is_atypique  = st.session_state.get("is_atypique", False)
        nrd_pds_ref  = 45 if pd_data["proc_type"] == "CORO" else 85
        pct_nrd      = pds / nrd_pds_ref * 100

        # Bannière succès / atypique
        if is_atypique:
            st.markdown(
                '<div class="alert-critique">⚠️ <b>Patient ATYPIQUE détecté (IsolationForest)</b> — '
                'Profil dosimétrique inhabituel par rapport aux 160 cas réels. '
                'Vérification recommandée.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.success("✅ Prédiction calculée — patient ajouté au dataset du service.")

        # KPI cards
        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        for col, (label, val, color, sub) in zip(
            [kc1, kc2, kc3, kc4, kc5],
            [
                ("PDS Total",     f"{pds:.1f} Gy·cm²",  "#a78bfa", f"{pct_nrd:.0f}% du NRD"),
                ("Kérma Air",     f"{kerma:.0f} mGy",   "#f59e0b", "Cumulé"),
                ("Dose Efficace", f"{dose_eff:.2f} mSv","#34d399", "Estimation"),
                ("Dose Peau",     f"{dose_peau:.2f} Gy","#22d3ee", "Pic"),
                ("% NRD",         f"{pct_nrd:.0f}%",
                 "#f87171" if pct_nrd > 100 else "#4ade80", pd_data["proc_type"]),
            ]
        ):
            col.markdown(f"""
            <div class="kpi-card" style="--kpi-color:{color}">
              <div class="kpi-value" style="color:{color}">{val}</div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown(f"""
            <div style="text-align:center;padding:14px">
              <div style="font-size:12px;color:#64748b;margin-bottom:8px;letter-spacing:0.08em">
                ÉVALUATION DU RISQUE
              </div>
              <div class="risk-badge risk-badge-{classe}">
                {risk['icon']} {risk['label'].upper()}
              </div>
              <div style="font-size:13px;color:#94a3b8;margin-top:14px;max-width:340px;
                          margin-left:auto;margin-right:auto">
                {risk['desc']}
              </div>
            </div>""", unsafe_allow_html=True)

            # Cluster badge
            cname = CLUSTER_NAMES[cluster_num]
            ccolor = CLUSTER_COLORS[cluster_num]
            cbg    = CLUSTER_BG[cluster_num]
            st.markdown(f"""
            <div class="metric-box" style="margin-top:8px;border-color:{ccolor}40;">
              <div style="font-size:11px;color:#64748b;margin-bottom:4px">CLUSTER K-MEANS</div>
              <div style="font-size:18px;font-weight:800;color:{ccolor}">
                Cluster {cluster_num} — {cname}
              </div>
            </div>""", unsafe_allow_html=True)

            # Percentile
            pct_rank = (st.session_state.main_dataset["PDS_TOTAL"] < pds).mean() * 100
            st.markdown(f"""
            <div class="metric-box" style="margin-top:10px">
              <div style="font-size:12px;color:#64748b">Percentile cohorte</div>
              <div style="font-size:22px;font-weight:800;color:#22d3ee">{pct_rank:.0f}e</div>
              <div style="font-size:11px;color:#475569">
                Plus irradié que {pct_rank:.0f}% des patients
              </div>
            </div>""", unsafe_allow_html=True)

        with c2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pds,
                delta={"reference": nrd_pds_ref, "valueformat": ".1f"},
                title={"text": f"PDS vs NRD {pd_data['proc_type']} ({nrd_pds_ref} Gy·cm²)"},
                gauge={
                    "axis": {"range": [0, nrd_pds_ref * 3]},
                    "bar": {"color": risk["color"]},
                    "steps": [
                        {"range": [0,              nrd_pds_ref],       "color": "#052e16"},
                        {"range": [nrd_pds_ref,    nrd_pds_ref * 1.5], "color": "#3f2d06"},
                        {"range": [nrd_pds_ref*1.5, nrd_pds_ref * 2],  "color": "#422006"},
                        {"range": [nrd_pds_ref*2,  nrd_pds_ref * 3],   "color": "#450a0a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": nrd_pds_ref},
                },
                number={"suffix": " Gy·cm²", "valueformat": ".1f"},
            ))
            fig_g.update_layout(paper_bgcolor="#0a1628", font_color="#94a3b8",
                                height=260, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_g, use_container_width=True)

        # Recommandations
        st.markdown('<div class="section-title">💡 Recommandations Cliniques</div>',
                    unsafe_allow_html=True)
        recs = get_recommendations(pds, kerma, pd_data["duree"], pd_data["imc"],
                                   pd_data["proc_type"], pd_data["nb_images"])
        for prio, msg in recs:
            level = ("critique" if "URGENT" in prio
                     else "danger" if "ÉLEVÉ" in prio
                     else "warning" if "MOYEN" in prio else "ok")
            st.markdown(f'<div class="alert-{level}"><b>{prio}</b>  {msg}</div>',
                        unsafe_allow_html=True)

        # ── Boutons d'action ────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

        with btn_col1:
            gen_pdf = st.button("📄 Générer Rapport PDF", use_container_width=True)

        with btn_col2:
            if st.button("🔄 Nouvelle Prédiction", use_container_width=True):
                st.session_state.pred_pds       = None
                st.session_state.pred_kerma     = None
                st.session_state.pred_dose_eff  = None
                st.session_state.pred_dose_peau = None
                st.session_state.classe         = None
                st.session_state.patient_data   = None
                st.session_state.show_new_pred_hint = True
                st.rerun()

        if gen_pdf:
            with st.spinner("📄 Génération du rapport PDF…"):
                pdf_bytes = generate_pdf_report(
                    patient_data=pd_data,
                    pds=pds, kerma=kerma, dose_eff=dose_eff, dose_peau=dose_peau,
                    classe=classe, recs=recs,
                    cluster_num=cluster_num,
                    cluster_name=CLUSTER_NAMES[cluster_num],
                    pct_rank=pct_rank,
                    main_dataset=st.session_state.main_dataset,
                )
            if pdf_bytes is None:
                st.warning("⚠️ ReportLab non installé. Exécutez : `pip install reportlab`")
            else:
                filename = f"CardioDose_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button("⬇️ Télécharger le Rapport PDF", pdf_bytes,
                                   filename, "application/pdf",
                                   use_container_width=True)
                st.success("✅ Rapport PDF généré avec succès !")

        if st.session_state.show_new_pred_hint:
            st.markdown('<div class="new-pred-hint">👆 Modifiez les paramètres dans la barre '
                        'latérale puis cliquez sur <b>PRÉDIRE</b> pour une nouvelle analyse.</div>',
                        unsafe_allow_html=True)

        # ── Explicabilité SHAP (optionnel) ──────────────────────────────────
        with st.expander("🔬 Détail technique — Facteurs influençant cette dose (SHAP)"):
            try:
                import shap
                sexe_bin_ = 1 if pd_data["sexe"] == "HOMME" else 0
                proc_bin_ = 1 if pd_data["proc_type"] == "ANGIO" else 0
                X_shap = pd.DataFrame(
                    [[pd_data["age"], sexe_bin_, pd_data["imc"], pd_data["duree"],
                      pd_data["nb_series"], pd_data["nb_images"], proc_bin_]],
                    columns=RF_FEATURES,
                )
                explainer  = shap.TreeExplainer(models["PDS_TOTAL"])
                shap_vals  = explainer.shap_values(X_shap)
                sv = shap_vals[0]
                feat_fr  = ["Âge", "Sexe", "IMC", "Durée scopie", "Nb séries", "Nb images", "Procédure"]
                sorted_i = np.argsort(np.abs(sv))[::-1]
                fig_shap = go.Figure(go.Bar(
                    x=[sv[i] for i in sorted_i],
                    y=[feat_fr[i] for i in sorted_i],
                    orientation="h",
                    marker_color=["#f87171" if sv[i] > 0 else "#4ade80" for i in sorted_i],
                ))
                fig_shap.update_layout(
                    title="Impact sur le PDS (rouge = augmente, vert = diminue)",
                    template="plotly_dark", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
                    height=280, xaxis_title="Effet sur PDS (Gy·cm²)",
                    margin=dict(t=40, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            except ImportError:
                st.info("Installez `shap` pour l'explicabilité détaillée : `pip install shap`")
            except Exception as e:
                st.warning(f"SHAP non disponible pour ce calcul : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 2 — MES SIMULATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-title">📊 Historique de mes Simulations (session en cours)</div>',
                unsafe_allow_html=True)

    history = st.session_state.predictions_history
    if not history:
        st.markdown('<div class="alert-info">ℹ️ Aucune simulation enregistrée. '
                    'Allez dans <b>🏠 Accueil & Prédiction</b> pour simuler votre premier patient.</div>',
                    unsafe_allow_html=True)
    else:
        hist_df = pd.DataFrame(history)

        n_total = len(hist_df)
        n_risk  = hist_df["Classe"].isin(["C", "D"]).sum()
        pds_moy = hist_df["PDS (Gy·cm²)"].mean()

        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, (label, val, color, sub) in zip(
            [sc1, sc2, sc3, sc4],
            [
                ("Simulations",            str(n_total),          "#22d3ee", "cette session"),
                ("PDS moyen simulé",       f"{pds_moy:.1f} Gy·cm²","#a78bfa", ""),
                ("À risque élevé/critique",str(n_risk),
                 "#f87171" if n_risk > 0 else "#4ade80",
                 f"{n_risk/n_total*100:.0f}% des simulations"),
                ("Dernière simulation",    hist_df["Heure"].iloc[-1], "#34d399", ""),
            ]
        ):
            col.markdown(f"""
            <div class="kpi-card" style="--kpi-color:{color}">
              <div class="kpi-value" style="color:{color}">{val}</div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Évolution PDS simulé
        hist_df["#"] = range(1, len(hist_df) + 1)
        fig_evol = px.bar(
            hist_df, x="#", y="PDS (Gy·cm²)", color="Classe",
            color_discrete_map={c: CLINICAL_RISK[c]["color"] for c in CLINICAL_RISK},
            title="Évolution du PDS prédit au fil de mes simulations",
            template="plotly_dark",
            hover_data=["Heure", "Sexe", "Âge", "Procédure", "Cluster"],
        )
        fig_evol.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
                               height=320, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_evol, use_container_width=True)

        # Répartition des clusters simulés
        if "Cluster" in hist_df.columns:
            fig_cl_pie = px.pie(
                hist_df, names=hist_df["Cluster"].map(CLUSTER_NAMES),
                title="Répartition des clusters — patients simulés",
                template="plotly_dark",
                color_discrete_sequence=list(CLUSTER_COLORS.values()),
            )
            fig_cl_pie.update_layout(paper_bgcolor="#0a1628", height=300,
                                     margin=dict(t=40, b=10, l=10, r=10))
            col_cp, col_dummy = st.columns([1, 1])
            with col_cp:
                st.plotly_chart(fig_cl_pie, use_container_width=True)

        st.markdown('<div class="section-title">🗃️ Détail de mes simulations</div>',
                    unsafe_allow_html=True)
        display_df = hist_df.drop(columns=["#"]).copy()
        st.dataframe(
            display_df.style.background_gradient(
                subset=["PDS (Gy·cm²)", "Kérma (mGy)"], cmap="YlOrRd"),
            use_container_width=True,
        )

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.download_button(
                "⬇️ Exporter mes simulations (CSV)",
                display_df.to_csv(index=False).encode("utf-8"),
                "mes_simulations.csv", "text/csv", use_container_width=True,
            )
        with bcol2:
            if st.button("🗑️ Réinitialiser mes simulations", use_container_width=True):
                st.session_state.predictions_history = []
                st.session_state.main_dataset        = df_real_main.copy()
                st.session_state.pred_pds            = None
                st.session_state.classe              = None
                st.success("Historique réinitialisé.")
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 1 — RAPPORT PDF
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:

    # ── En-tête de l'onglet ───────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#0d2137;border:1px solid #1e3a5f;border-radius:12px;
                padding:18px 24px;margin-bottom:20px;display:flex;align-items:center;gap:16px;">
      <div style="font-size:40px">📄</div>
      <div>
        <div style="font-size:16px;font-weight:800;color:#22d3ee">
          Rapport de Radioprotection — CHU Mustapha Bacha Alger
        </div>
        <div style="font-size:12px;color:#64748b;margin-top:4px">
          Générez un rapport PDF complet et professionnel pour le dossier médical du patient.
          Le rapport intègre le logo officiel du CHU, les résultats dosimétriques,
          le profil clinique, les recommandations et la conformité aux NRD internationaux.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.pred_pds is None:
        # Pas encore de prédiction — afficher un guide visuel
        st.markdown("""
        <div class="alert-info">
          ℹ️ <b>Aucune prédiction disponible.</b><br>
          Pour générer un rapport, commencez par renseigner les paramètres du patient dans la
          barre latérale et cliquez sur <b>🔮 PRÉDIRE LA DOSE</b> dans l'onglet
          <b>🏠 Accueil & Prédiction</b>.
        </div>
        """, unsafe_allow_html=True)

        # Présentation des sections du rapport
        st.markdown('<div class="section-title">📋 Contenu du Rapport PDF</div>',
                    unsafe_allow_html=True)
        sections = [
            ("🏥", "En-tête officiel CHU",
             "Logo CHU Mustapha Bacha Alger, titre du service, date et heure de génération."),
            ("👤", "Informations Patient & Procédure",
             "Données démographiques, type de procédure, durée, comorbidités."),
            ("📐", "Résultats de la Prédiction",
             "PDS Total, Kérma Air, Dose Efficace, Dose Peau — avec comparaison au NRD de référence."),
            ("🚦", "Évaluation du Risque",
             "Classification ICRP (A/B/C/D), pourcentage du NRD, position dans la cohorte (percentile)."),
            ("👥", "Profil Clinique du Patient",
             "Groupe de similitude parmi les 4 profils de la cohorte, avec explication clinique et recommandation."),
            ("💡", "Recommandations Personnalisées",
             "Actions prioritaires classées par niveau d'urgence (URGENT / ÉLEVÉ / MOYEN / OK)."),
            ("🌍", "Conformité NRD Internationaux",
             "Tableau comparatif France ASN, ESC Europe, IAEA, Algérie, Belgique, Royaume-Uni."),
            ("📊", "Statistiques de la Cohorte",
             "Contexte du service : PDS médian, Kérma médian, taux de dépassement NRD."),
        ]
        for icon, titre, desc in sections:
            st.markdown(f"""
            <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;
                        padding:12px 16px;margin-bottom:8px;display:flex;gap:14px;align-items:flex-start;">
              <div style="font-size:22px;margin-top:2px">{icon}</div>
              <div>
                <div style="font-size:13px;font-weight:700;color:#cbd5e1">{titre}</div>
                <div style="font-size:12px;color:#64748b;margin-top:2px">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── Prédiction disponible — afficher récapitulatif + bouton génération ──
        pds_r       = st.session_state.pred_pds
        kerma_r     = st.session_state.pred_kerma
        dose_eff_r  = st.session_state.pred_dose_eff
        dose_peau_r = st.session_state.pred_dose_peau
        classe_r    = st.session_state.classe
        pd_data_r   = st.session_state.patient_data
        cluster_r   = st.session_state.get("cluster_num", 0)
        risk_r      = CLINICAL_RISK[classe_r]
        nrd_pds_r   = 45 if pd_data_r["proc_type"] == "CORO" else 85
        pct_nrd_r   = pds_r / nrd_pds_r * 100
        pct_rank_r  = (st.session_state.main_dataset["PDS_TOTAL"] < pds_r).mean() * 100
        recs_r      = get_recommendations(pds_r, kerma_r, pd_data_r["duree"],
                                          pd_data_r["imc"], pd_data_r["proc_type"],
                                          pd_data_r["nb_images"])

        # ── Aperçu du rapport ──────────────────────────────────────────────────
        st.markdown('<div class="section-title">👁️ Aperçu du Rapport à Générer</div>',
                    unsafe_allow_html=True)

        # En-tête simulé (bannière bleue imitant le PDF)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1b2a,#0f2a45);
                    border:1px solid #1e3a5f;border-radius:12px;
                    padding:20px 24px;margin-bottom:16px;">
          <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;">
            <div style="width:52px;height:52px;background:#1e3a5f;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:26px;border:2px solid #22d3ee;">🏥</div>
            <div>
              <div style="font-size:18px;font-weight:800;color:#22d3ee">
                CardioDose Optimizer
              </div>
              <div style="font-size:12px;color:#64748b">
                Rapport de Radioprotection — CHU Mustapha Bacha Alger — Cardiologie Interventionnelle
              </div>
              <div style="font-size:11px;color:#334155;margin-top:2px">
                Généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}
              </div>
            </div>
          </div>
          <div style="border-top:1px solid #1e3a5f;padding-top:12px;">
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;">
              <div style="text-align:center">
                <div style="font-size:10px;color:#475569;text-transform:uppercase;
                            letter-spacing:0.08em">Patient</div>
                <div style="font-size:14px;font-weight:700;color:#cbd5e1;margin-top:3px">
                  {pd_data_r['sexe']}, {pd_data_r['age']} ans
                </div>
              </div>
              <div style="text-align:center">
                <div style="font-size:10px;color:#475569;text-transform:uppercase;
                            letter-spacing:0.08em">Procédure</div>
                <div style="font-size:14px;font-weight:700;color:#cbd5e1;margin-top:3px">
                  {pd_data_r['proc_type']}
                </div>
              </div>
              <div style="text-align:center">
                <div style="font-size:10px;color:#475569;text-transform:uppercase;
                            letter-spacing:0.08em">Classe ICRP</div>
                <div style="font-size:14px;font-weight:700;color:{risk_r['color']};margin-top:3px">
                  {risk_r['icon']} {risk_r['label']}
                </div>
              </div>
              <div style="text-align:center">
                <div style="font-size:10px;color:#475569;text-transform:uppercase;
                            letter-spacing:0.08em">Profil clinique</div>
                <div style="font-size:14px;font-weight:700;
                            color:{CLUSTER_COLORS[cluster_r]};margin-top:3px">
                  {CLUSTER_NAMES[cluster_r]}
                </div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI récapitulatifs ─────────────────────────────────────────────────
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        for col, (label, val, color, sub) in zip(
            [rc1, rc2, rc3, rc4, rc5],
            [
                ("PDS Total",      f"{pds_r:.1f} Gy·cm²",   "#a78bfa", f"{pct_nrd_r:.0f}% du NRD"),
                ("Kérma Air",      f"{kerma_r:.0f} mGy",     "#f59e0b", "Cumulé"),
                ("Dose Efficace",  f"{dose_eff_r:.2f} mSv",  "#34d399", "Estimation"),
                ("Dose Peau",      f"{dose_peau_r:.2f} Gy",  "#22d3ee", "Pic"),
                ("Percentile",     f"{pct_rank_r:.0f}e",
                 "#f87171" if pct_rank_r > 75 else "#4ade80", "dans la cohorte"),
            ]
        ):
            col.markdown(f"""
            <div class="kpi-card" style="--kpi-color:{color}">
              <div class="kpi-value" style="color:{color};font-size:18px">{val}</div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Recommandations à inclure ──────────────────────────────────────────
        st.markdown('<div class="section-title">💡 Recommandations qui figureront dans le rapport</div>',
                    unsafe_allow_html=True)
        for prio_r, msg_r in recs_r:
            level_r = ("critique" if "URGENT" in prio_r
                       else "danger" if "ÉLEVÉ" in prio_r
                       else "warning" if "MOYEN" in prio_r else "ok")
            st.markdown(f'<div class="alert-{level_r}"><b>{prio_r}</b>  {msg_r}</div>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gros bouton de génération ──────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;font-size:13px;color:#64748b;margin-bottom:8px">
          ↓ Cliquez pour générer et télécharger le rapport complet (format PDF / A4)
        </div>""", unsafe_allow_html=True)

        gen_pdf_tab = st.button("📄  GÉNÉRER LE RAPPORT PDF OFFICIEL",
                                type="primary", use_container_width=True,
                                key="gen_pdf_tab2")

        if gen_pdf_tab:
            with st.spinner("📄 Génération du rapport PDF en cours… (logo, tableaux, recommandations)"):
                pdf_bytes_tab = generate_pdf_report(
                    patient_data=pd_data_r,
                    pds=pds_r, kerma=kerma_r,
                    dose_eff=dose_eff_r, dose_peau=dose_peau_r,
                    classe=classe_r, recs=recs_r,
                    cluster_num=cluster_r,
                    cluster_name=CLUSTER_NAMES[cluster_r],
                    pct_rank=pct_rank_r,
                    main_dataset=st.session_state.main_dataset,
                )
            if pdf_bytes_tab is None:
                st.error("❌ ReportLab n'est pas installé sur ce serveur.")
                st.code("pip install reportlab", language="bash")
            else:
                fname_tab = (f"CardioDose_Rapport_"
                             f"{pd_data_r['sexe']}_{pd_data_r['age']}ans_"
                             f"{pd_data_r['proc_type']}_"
                             f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
                st.download_button(
                    "⬇️  TÉLÉCHARGER LE RAPPORT PDF",
                    pdf_bytes_tab,
                    fname_tab,
                    "application/pdf",
                    use_container_width=True,
                )
                st.success(f"✅ Rapport généré avec succès ! Fichier : {fname_tab}")
                st.markdown("""
                <div style="background:#052e16;border:1px solid #166534;border-radius:8px;
                            padding:12px 16px;margin-top:10px;font-size:12px;color:#4ade80;">
                  📌 <b>Rappel :</b> Ce rapport est un outil d'aide à la décision clinique.
                  Il ne remplace pas l'évaluation médicale qualifiée du radiologue ou du cardiologue référent.
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 3 — DONNÉES DU SERVICE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    main_df = st.session_state.main_dataset
    n_real  = (main_df["SOURCE"] == "Réel").sum()
    n_sim   = (main_df["SOURCE"] == "Simulé").sum()

    st.markdown(f'<div class="section-title">📈 Statistiques du Service — '
                f'{n_real} cas réels + {n_sim} simulés = {len(main_df)} patients</div>',
                unsafe_allow_html=True)

    # ── 🕒 Contrôle temporel ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">🕒 Contrôle Temporel</div>', unsafe_allow_html=True)
    st.caption("ℹ️ Le fichier source ne contient pas de date d'examen native : une date est "
               "reconstituée de façon déterministe (répartition chronologique sur les 12 derniers "
               "mois, dans l'ordre des patients) afin de permettre l'analyse temporelle ci-dessous.")

    main_df_dated = main_df.reset_index(drop=True).copy()
    _date_end   = pd.Timestamp.now().normalize()
    _date_start = _date_end - pd.Timedelta(days=364)
    main_df_dated["DATE_EXAM"] = pd.date_range(_date_start, _date_end, periods=len(main_df_dated))

    periode = st.radio("Période d'analyse", ["Jour", "Semaine", "Mois", "Année"],
                       horizontal=True, key="periode_service")
    _lookback = {"Jour": 1, "Semaine": 7, "Mois": 30, "Année": 365}[periode]
    _cutoff   = _date_end - pd.Timedelta(days=_lookback)
    main_df_f = main_df_dated[main_df_dated["DATE_EXAM"] >= _cutoff]
    if main_df_f.empty:
        st.markdown('<div class="alert-warning">⚠️ Aucun patient sur cette période — '
                    'affichage de la cohorte complète à la place.</div>', unsafe_allow_html=True)
        main_df_f = main_df_dated
    st.caption(f"📌 {len(main_df_f)} patient(s) sur la période sélectionnée "
               f"({periode.lower()}), sur un total de {len(main_df_dated)}.")

    nrd_ref_map = {"CORO": 45, "ANGIO": 85}
    pct_dep     = main_df.apply(
        lambda r: r["PDS_TOTAL"] > nrd_ref_map.get(r["PROC_TYPE"], 45), axis=1).mean() * 100
    pct_k2000   = (main_df["KERMA_mGy"] > 2000).mean() * 100

    kpis = [
        ("👥", "Patients (total)",       f"{len(main_df)}",
         "#22d3ee", f"{n_real} réels + {n_sim} simulés"),
        ("📐", "PDS Médian",             f"{main_df['PDS_TOTAL'].median():.1f} Gy·cm²",
         "#a78bfa", ""),
        ("⚡", "Kérma Médian",           f"{main_df['KERMA_mGy'].median():.0f} mGy",
         "#f59e0b", ""),
        ("💊", "Dose Eff. Médiane",      f"{main_df['DOSE_EFF'].median():.2f} mSv",
         "#34d399", ""),
        ("🚨", "% > NRD France",         f"{pct_dep:.1f}%",  "#f87171", "Dépassement"),
        ("⚠️", "% Kérma > 2000",        f"{pct_k2000:.1f}%","#fb923c", "Surveillance dermato"),
    ]
    kcols = st.columns(6)
    for col, (icon, label, val, color, sub) in zip(kcols, kpis):
        col.markdown(f"""
        <div class="kpi-card" style="--kpi-color:{color}">
          <div style="font-size:22px">{icon}</div>
          <div class="kpi-value" style="color:{color}">{val}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Histogramme + Boxplot ────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig_h = px.histogram(main_df, x="PDS_TOTAL", nbins=30, color="PROC_TYPE",
                             title="Distribution PDS Total (Gy·cm²)",
                             template="plotly_dark",
                             color_discrete_map={"CORO": "#22d3ee", "ANGIO": "#a78bfa"})
        fig_h.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#0a1628")
        st.plotly_chart(fig_h, use_container_width=True)
    with c2:
        fig_box = px.box(main_df, x="PROC_TYPE", y="KERMA_mGy", color="PROC_TYPE",
                         points="all",
                         title="Kérma par Type de Procédure — Boxplot (mGy)",
                         template="plotly_dark",
                         color_discrete_map={"CORO": "#22d3ee", "ANGIO": "#a78bfa"})
        fig_box.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#0a1628")
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Violin plots ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎻 Violin Plots — Distribution par variable dosimétrique</div>',
                unsafe_allow_html=True)

    viol_var = st.selectbox(
        "Variable à afficher",
        ["PDS_TOTAL", "KERMA_mGy", "DOSE_EFF", "DOSE_PEAU", "DUREE_MIN", "NB_SERIES"],
        format_func=lambda x: {
            "PDS_TOTAL": "PDS Total (Gy·cm²)", "KERMA_mGy": "Kérma (mGy)",
            "DOSE_EFF": "Dose Efficace (mSv)", "DOSE_PEAU": "Dose Peau (Gy)",
            "DUREE_MIN": "Durée Scopie (min)", "NB_SERIES": "Nombre de Séries",
        }.get(x, x),
    )
    vc1, vc2 = st.columns(2)
    with vc1:
        fig_viol = px.violin(main_df, x="PROC_TYPE", y=viol_var, color="PROC_TYPE",
                             box=True, points="all",
                             title=f"Distribution de {viol_var} par type de procédure",
                             template="plotly_dark",
                             color_discrete_map={"CORO": "#22d3ee", "ANGIO": "#a78bfa"})
        fig_viol.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#0a1628", height=380)
        st.plotly_chart(fig_viol, use_container_width=True)
    with vc2:
        fig_viol2 = px.violin(main_df, x="SEXE", y=viol_var, color="SEXE",
                              box=True, points="all",
                              title=f"Distribution de {viol_var} par sexe",
                              template="plotly_dark",
                              color_discrete_map={"FEMME": "#f472b6", "HOMME": "#60a5fa"})
        fig_viol2.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#0a1628", height=380)
        st.plotly_chart(fig_viol2, use_container_width=True)

    # ── Scatter plots ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📉 Corrélations clés</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        fig_sc = px.scatter(main_df, x="DUREE_MIN", y="PDS_TOTAL", color="PROC_TYPE",
                            symbol="SOURCE", trendline="ols",
                            title="PDS vs Durée Scopie (◆ = simulé)",
                            template="plotly_dark",
                            color_discrete_map={"CORO": "#22d3ee", "ANGIO": "#a78bfa"})
        fig_sc.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#0a1628")
        st.plotly_chart(fig_sc, use_container_width=True)
    with c4:
        fig_sc2 = px.scatter(main_df, x="IMC", y="PDS_TOTAL", color="PROC_TYPE",
                             symbol="SOURCE", trendline="ols",
                             title="PDS vs IMC (◆ = simulé)",
                             template="plotly_dark",
                             color_discrete_map={"CORO": "#22d3ee", "ANGIO": "#a78bfa"})
        fig_sc2.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#0a1628")
        st.plotly_chart(fig_sc2, use_container_width=True)

    # ── Heatmap corrélations (Spearman) ──────────────────────────────────────
    st.markdown('<div class="section-title">🌡️ Heatmap Corrélations Spearman</div>',
                unsafe_allow_html=True)
    heat_cols = ["IMC", "DUREE_MIN", "NB_SERIES", "NB_IMAGES",
                 "PDS_TOTAL", "KERMA_mGy", "DOSE_EFF", "DOSE_PEAU"]
    corr = main_df[heat_cols].corr(method="spearman").round(2)
    fig_heat = px.imshow(
        corr, text_auto=True, template="plotly_dark",
        color_continuous_scale="RdBu_r",
        title="Corrélations Spearman — cohorte complète (réels + simulés)",
        zmin=-1, zmax=1,
    )
    fig_heat.update_layout(paper_bgcolor="#0a1628", height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Performances Random Forest ────────────────────────────────────────────
    st.markdown('<div class="section-title">🤖 Performances Random Forest '
                '(160 cas réels)</div>', unsafe_allow_html=True)
    labels_fr = {"PDS_TOTAL": "PDS Total", "KERMA_mGy": "Kérma",
                 "DOSE_EFF": "Dose Efficace", "DOSE_PEAU": "Dose Peau"}
    mc = st.columns(4)
    for col, (t, lbl) in zip(mc, labels_fr.items()):
        m = rf_metrics[t]
        col.markdown(f"""
        <div class="kpi-card" style="--kpi-color:#22d3ee">
          <div style="font-size:11px;color:#64748b;margin-bottom:6px">{lbl}</div>
          <div class="kpi-value" style="color:#22d3ee">R²={m['R²']}</div>
          <div class="kpi-sub">MAE={m['MAE']} | RMSE={m['RMSE']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1:
        importances = models["PDS_TOTAL"].feature_importances_
        feat_fr = ["Âge", "Sexe", "IMC", "Durée scopie", "Nb séries", "Nb images", "Procédure"]
        sidx    = np.argsort(importances)
        fig_imp = go.Figure(go.Bar(
            x=importances[sidx], y=[feat_fr[i] for i in sidx],
            orientation="h", marker_color="#22d3ee",
        ))
        fig_imp.update_layout(template="plotly_dark", paper_bgcolor="#0a1628",
                              plot_bgcolor="#0a1628", title="Importance des variables (PDS)",
                              xaxis_title="Importance (Gini)", height=320,
                              margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_imp, use_container_width=True)
    with ic2:
        y_pred_all = models["PDS_TOTAL"].predict(df_real[RF_FEATURES])
        mx = max(df_real["PDS_TOTAL"].max(), y_pred_all.max())
        fig_pva = go.Figure()
        fig_pva.add_trace(go.Scatter(
            x=df_real["PDS_TOTAL"], y=y_pred_all, mode="markers",
            marker=dict(color="#a78bfa", size=5, opacity=0.7), name="Patients réels",
        ))
        fig_pva.add_trace(go.Scatter(
            x=[0, mx], y=[0, mx], mode="lines",
            line=dict(color="#f87171", dash="dash"), name="Prédiction parfaite",
        ))
        fig_pva.update_layout(template="plotly_dark", paper_bgcolor="#0a1628",
                              plot_bgcolor="#0a1628", title="PDS Réel vs Prédit",
                              xaxis_title="PDS Réel (Gy·cm²)", yaxis_title="PDS Prédit",
                              height=320, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_pva, use_container_width=True)

    # ── Tableau filtrable + exports ───────────────────────────────────────────
    st.markdown('<div class="section-title">🗃️ Données du Service (filtrable)</div>',
                unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    with fc1:
        proc_f = st.multiselect("Type de procédure", ["CORO", "ANGIO"],
                                default=["CORO", "ANGIO"])
    with fc2:
        src_f = st.multiselect("Origine", ["Réel", "Simulé"], default=["Réel", "Simulé"])
    df_show = main_df[main_df["PROC_TYPE"].isin(proc_f) & main_df["SOURCE"].isin(src_f)]
    st.dataframe(
        df_show.style.background_gradient(subset=["PDS_TOTAL", "KERMA_mGy"], cmap="YlOrRd"),
        use_container_width=True,
    )

    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button("⬇️ Exporter CSV (filtré)",
                           df_show.to_csv(index=False).encode("utf-8"),
                           "dataset_filtre.csv", "text/csv", use_container_width=True)
    with ec2:
        try:
            import openpyxl
            df_exp = assign_cluster_anomaly(
                main_df, cl_scaler, km_model, iso_model, pca_model, CLUSTER_REMAP)
            df_exp["NOM_CLUSTER"]    = df_exp["CLUSTER"].map(CLUSTER_NAMES)
            df_exp["STATUT_ANOMALIE"]= df_exp["ANOMALIE"].map({1: "Normal", -1: "Atypique"})
            df_exp = df_exp.drop(columns=["PC1", "PC2", "PC3"], errors="ignore")
            xl_buf = io.BytesIO()
            with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                df_exp.to_excel(writer, sheet_name="Dataset_Complet", index=False)
                pd.DataFrame(rf_metrics).T.reset_index().rename(
                    columns={"index": "Cible"}).to_excel(
                    writer, sheet_name="Metriques_IA", index=False)
            xl_buf.seek(0)
            st.download_button("⬇️ Exporter Excel Complet", xl_buf.getvalue(),
                               "cardiodose_export.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        except ImportError:
            st.caption("_Export Excel : `pip install openpyxl`_")

    with st.expander("🔬 Statistiques descriptives & qualité des données"):
        num_c = ["AGE", "IMC", "DUREE_MIN", "NB_IMAGES", "NB_SERIES",
                 "PDS_TOTAL", "KERMA_mGy", "DOSE_EFF"]
        dq1, dq2 = st.columns(2)
        with dq1:
            st.markdown("**Statistiques descriptives**")
            st.dataframe(main_df[num_c].describe().round(2), use_container_width=True)
        with dq2:
            st.markdown("**Qualité des données réelles (N=160)**")
            miss = df_real.isnull().sum()
            miss_df = miss[miss > 0].reset_index()
            miss_df.columns = ["Colonne", "Valeurs manquantes"]
            if miss_df.empty:
                st.success("✅ Aucune valeur manquante après imputation.")
            else:
                st.dataframe(miss_df, use_container_width=True)
            st.metric("Doublons (réels)", df_real.duplicated().sum())

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 5 — ANALYSE DES PROFILS CLINIQUES (anciennement Data Mining & Clustering)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    main_df = st.session_state.main_dataset

    # ── Introduction clinique ─────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#0d2137;border:1px solid #1e3a5f;border-radius:12px;
                padding:18px 22px;margin-bottom:18px;">
      <div style="font-size:15px;font-weight:700;color:#22d3ee;margin-bottom:6px;">
        🩺 Qu'est-ce que l'analyse des profils cliniques ?
      </div>
      <div style="font-size:13px;color:#94a3b8;line-height:1.8;">
        Cette analyse regroupe automatiquement les patients de la cohorte en <b style="color:#e2e8f0;">4 profils
        dosimétrique</b> selon leurs caractéristiques (IMC, durée de la procédure, nombre d'images/séries,
        dose reçue). L'objectif est d'identifier les patients similaires pour mieux
        adapter les recommandations et anticiper les risques radiologiques.<br>
        <span style="color:#64748b;font-size:11px;">
          Les profils sont figés sur les 160 cas réels du service. Chaque nouveau patient simulé
          est automatiquement comparé à ces profils de référence.
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Profil du patient en cours (si prédiction disponible) ─────────────────
    if st.session_state.pred_pds is not None:
        cur_cluster = st.session_state.get("cluster_num", 0)
        cur_color   = CLUSTER_COLORS[cur_cluster]
        cur_bg      = CLUSTER_BG[cur_cluster]
        cur_name    = CLUSTER_NAMES[cur_cluster]
        CLUSTER_CLINICAL_DESC = {
            0: ("Ce patient présente une exposition aux rayonnements inférieure à la moyenne du service. "
                "La durée de la procédure, le nombre d'images et la dose reçue sont tous modérés. "
                "Aucune action particulière n'est requise au-delà des bonnes pratiques habituelles."),
            1: ("Ce patient présente une exposition modérée, comparable à la majorité des patients "
                "de la cohorte. Une surveillance standard est recommandée. "
                "Vérifier que la durée de scopie reste dans les limites habituelles."),
            2: ("Ce patient présente une exposition supérieure à la moyenne. "
                "La procédure a nécessité plus d'acquisitions ou une durée plus longue que la majorité. "
                "Un audit dosimétrique est recommandé et une surveillance cutanée à J10–J14 est conseillée."),
            3: ("Ce patient présente une exposition critique, parmi les plus élevées de la cohorte. "
                "La dose reçue dépasse significativement les niveaux de référence. "
                "Un suivi dermatologique immédiat est obligatoire et un audit de la procédure est requis."),
        }
        st.markdown(f"""
        <div style="background:{cur_bg};border:2px solid {cur_color};border-radius:14px;
                    padding:20px 24px;margin-bottom:20px;">
          <div style="font-size:12px;color:#64748b;text-transform:uppercase;
                      letter-spacing:0.1em;margin-bottom:6px;">
            Profil du patient analysé
          </div>
          <div style="font-size:22px;font-weight:800;color:{cur_color};margin-bottom:10px;">
            {cur_name}
          </div>
          <div style="font-size:13px;color:#cbd5e1;line-height:1.7;">
            {CLUSTER_CLINICAL_DESC[cur_cluster]}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-info">ℹ️ Effectuez d'abord une prédiction (onglet Accueil) pour voir
        à quel profil appartient votre patient.</div>
        """, unsafe_allow_html=True)

    # ── Assignation clusters sur tout le dataset ──────────────────────────────
    df_assigned = assign_cluster_anomaly(
        main_df, cl_scaler, km_model, iso_model, pca_model, CLUSTER_REMAP)

    # ── Cartes de profils — langage clinique ───────────────────────────────────
    st.markdown('<div class="section-title">👥 Les 4 Profils de Patients de la Cohorte</div>',
                unsafe_allow_html=True)

    CLUSTER_RECO = {
        0: "Maintenir les bonnes pratiques. Aucune action dosimétrique spécifique.",
        1: "Surveillance standard. Vérifier la durée de scopie à chaque procédure.",
        2: "Audit dosimétrique recommandé. Surveillance cutanée J10–J14 si dose peau > 1 Gy.",
        3: "Suivi dermatologique obligatoire. Révision urgente du protocole de la procédure.",
    }

    col_cl = st.columns(4)
    for ci in range(4):
        sub = df_assigned[df_assigned["CLUSTER"] == ci]
        if len(sub) == 0:
            continue
        color = CLUSTER_COLORS[ci]
        bg    = CLUSTER_BG[ci]
        # Highlight si c'est le cluster du patient en cours
        border_width = "3px" if (st.session_state.pred_pds is not None and
                                  st.session_state.get("cluster_num", -1) == ci) else "1px"
        is_current   = (st.session_state.pred_pds is not None and
                        st.session_state.get("cluster_num", -1) == ci)
        marker_html  = f'<div style="font-size:10px;color:{color};font-weight:700;margin-bottom:4px">◀ PATIENT EN COURS</div>' if is_current else ""
        with col_cl[ci]:
            n_real_cl = (sub["SOURCE"] == "Réel").sum()
            n_sim_cl  = (sub["SOURCE"] == "Simulé").sum()
            st.markdown(f"""
            <div class="kpi-card" style="--kpi-color:{color};background:{bg};
                         border:{border_width} solid {color};">
              {marker_html}
              <div class="kpi-value" style="color:{color};font-size:16px;margin-bottom:8px">
                {CLUSTER_NAMES[ci]}
              </div>
              <div class="kpi-sub" style="font-size:11px;margin-bottom:10px">
                <b style="color:#cbd5e1">{len(sub)} patients</b>
                ({n_real_cl} réels · {n_sim_cl} simulés)
              </div>
              <div class="explain-item" style="font-size:11px;color:#94a3b8;padding:4px 0;
                            border-bottom:1px solid #0f1f33">
                📐 Dose moy. : <b style="color:{color}">{sub['PDS_TOTAL'].mean():.1f} Gy·cm²</b>
              </div>
              <div class="explain-item" style="font-size:11px;color:#94a3b8;padding:4px 0;
                            border-bottom:1px solid #0f1f33">
                ⚡ Kérma moy. : <b style="color:{color}">{sub['KERMA_mGy'].mean():.0f} mGy</b>
              </div>
              <div class="explain-item" style="font-size:11px;color:#94a3b8;padding:4px 0;
                            border-bottom:1px solid #0f1f33">
                ⏱️ Durée moy. : <b style="color:{color}">{sub['DUREE_MIN'].mean():.1f} min</b>
              </div>
              <div class="explain-item" style="font-size:11px;color:#94a3b8;padding:4px 0;
                            border-bottom:1px solid #0f1f33">
                ⚖️ IMC moy. : <b style="color:{color}">{sub['IMC'].mean():.1f} kg/m²</b>
              </div>
              <div style="margin-top:8px;font-size:10px;color:#64748b;line-height:1.5">
                💡 {CLUSTER_RECO[ci]}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparaison du patient avec les autres ────────────────────────────────
    if st.session_state.pred_pds is not None:
        st.markdown('<div class="section-title">📊 Comparaison de ce Patient avec les Patients Similaires</div>',
                    unsafe_allow_html=True)
        cur_cluster  = st.session_state.get("cluster_num", 0)
        sub_similar  = df_assigned[df_assigned["CLUSTER"] == cur_cluster]
        pds_patient  = st.session_state.pred_pds
        kerma_patient = st.session_state.pred_kerma

        pct_pds   = (sub_similar["PDS_TOTAL"]  < pds_patient).mean()   * 100
        pct_kerma = (sub_similar["KERMA_mGy"] < kerma_patient).mean() * 100

        cmp1, cmp2, cmp3 = st.columns(3)
        cmp1.markdown(f"""
        <div class="metric-box">
          <div style="font-size:11px;color:#64748b">Patients dans ce profil</div>
          <div style="font-size:26px;font-weight:800;color:{CLUSTER_COLORS[cur_cluster]}">
            {len(sub_similar)}
          </div>
          <div style="font-size:11px;color:#475569">patients similaires</div>
        </div>""", unsafe_allow_html=True)
        cmp2.markdown(f"""
        <div class="metric-box">
          <div style="font-size:11px;color:#64748b">Position Dose (PDS)</div>
          <div style="font-size:26px;font-weight:800;color:#22d3ee">{pct_pds:.0f}e</div>
          <div style="font-size:11px;color:#475569">percentile dans le profil</div>
        </div>""", unsafe_allow_html=True)
        cmp3.markdown(f"""
        <div class="metric-box">
          <div style="font-size:11px;color:#64748b">Position Kérma</div>
          <div style="font-size:26px;font-weight:800;color:#f59e0b">{pct_kerma:.0f}e</div>
          <div style="font-size:11px;color:#475569">percentile dans le profil</div>
        </div>""", unsafe_allow_html=True)

        # Boxplot comparatif
        st.markdown("<br>", unsafe_allow_html=True)
        fig_comp = go.Figure()
        for ci_cmp in range(4):
            s = df_assigned[df_assigned["CLUSTER"] == ci_cmp]["PDS_TOTAL"]
            if len(s) == 0:
                continue
            fig_comp.add_trace(go.Box(
                y=s, name=CLUSTER_NAMES[ci_cmp],
                marker_color=CLUSTER_COLORS[ci_cmp], boxpoints="outliers",
                line=dict(color=CLUSTER_COLORS[ci_cmp]),
            ))
        fig_comp.add_hline(y=pds_patient, line_dash="dash", line_color="#22d3ee", line_width=2,
                           annotation_text=f"Votre patient : {pds_patient:.1f} Gy·cm²",
                           annotation_font_color="#22d3ee")
        fig_comp.update_layout(
            template="plotly_dark", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
            title="PDS Total — Comparaison du patient avec les 4 profils de la cohorte",
            yaxis_title="PDS Total (Gy·cm²)",
            height=380, margin=dict(t=50, b=10, l=10, r=10),
            showlegend=True,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── Radar chart par profil ────────────────────────────────────────────────
    st.markdown('<div class="section-title">🕸️ Comparaison des Profils — Diagramme en Étoile</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:#64748b;margin-bottom:12px;line-height:1.6">
      Ce diagramme compare les 4 profils selon 6 paramètres cliniques normalisés.
      Un profil plus étendu vers l'extérieur indique une exposition et une complexité procédurale plus élevées.
    </div>""", unsafe_allow_html=True)

    radar_cols   = ["PDS_TOTAL", "KERMA_mGy", "DUREE_MIN", "NB_SERIES", "NB_IMAGES", "IMC"]
    radar_labels = ["Dose (PDS)", "Kérma", "Durée scopie", "Nb séries", "Nb images", "IMC"]
    df_cl_norm   = df_assigned.groupby("CLUSTER")[radar_cols].mean()
    df_cl_norm   = (df_cl_norm - df_cl_norm.min()) / (df_cl_norm.max() - df_cl_norm.min() + 1e-9)

    fig_radar = go.Figure()
    for ci in range(4):
        if ci not in df_cl_norm.index:
            continue
        vals = df_cl_norm.loc[ci].tolist()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", name=CLUSTER_NAMES[ci],
            line=dict(color=CLUSTER_COLORS[ci]),
            fillcolor=CLUSTER_COLORS[ci], opacity=0.25,
        ))
    fig_radar.update_layout(
        template="plotly_dark", paper_bgcolor="#0a1628",
        polar=dict(bgcolor="#0a1628"),
        title="Empreinte clinique normalisée par profil",
        height=430, margin=dict(t=50, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Nuage de points interactif ─────────────────────────────────────────────
    st.markdown('<div class="section-title">🔵 Répartition Spatiale des Patients par Profil</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:#64748b;margin-bottom:12px;line-height:1.6">
      Chaque point représente un patient. Les patients proches sont cliniquement similaires.
      Survolez un point pour voir ses données. Les formes carrées (◆) représentent les patients simulés.
    </div>""", unsafe_allow_html=True)

    pca2d_df = df_assigned.copy()
    pca2d_df["Profil"] = pca2d_df["CLUSTER"].map(CLUSTER_NAMES)
    fig_pca2 = px.scatter(
        pca2d_df, x="PC1", y="PC2", color="Profil", symbol="SOURCE",
        hover_data=["PDS_TOTAL", "KERMA_mGy", "IMC"],
        title=(f"Carte des patients par profil clinique "
               f"(axes représentant {(pca_var_exp[0]+pca_var_exp[1])*100:.1f}% de la variabilité totale)"),
        template="plotly_dark",
        color_discrete_sequence=list(CLUSTER_COLORS.values()),
        labels={"PC1": "Axe principal 1", "PC2": "Axe principal 2",
                "Profil": "Profil clinique", "SOURCE": "Origine"},
    )
    fig_pca2.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
                           height=450, margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_pca2, use_container_width=True)

    # ── Vue 3D interactive ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🌐 Vue 3D des Profils — Rotation Interactive</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:#64748b;margin-bottom:12px">
      Vue tridimensionnelle — faites tourner le graphique pour mieux visualiser la séparation entre profils.
    </div>""", unsafe_allow_html=True)

    fig_pca3 = px.scatter_3d(
        pca2d_df, x="PC1", y="PC2", z="PC3",
        color="Profil", symbol="SOURCE",
        hover_data=["PDS_TOTAL", "KERMA_mGy", "IMC"],
        title=(f"Vue 3D — {pca_var_exp[0]*100:.1f}% | {pca_var_exp[1]*100:.1f}% | "
               f"{pca_var_exp[2]*100:.1f}% de la variabilité expliquée"),
        template="plotly_dark",
        color_discrete_sequence=list(CLUSTER_COLORS.values()),
        labels={"PC1": "Axe 1", "PC2": "Axe 2", "PC3": "Axe 3",
                "Profil": "Profil clinique", "SOURCE": "Origine"},
    )
    fig_pca3.update_layout(paper_bgcolor="#0a1628", height=500,
                           margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_pca3, use_container_width=True)

    # ── Tableau de synthèse des profils ───────────────────────────────────────
    st.markdown('<div class="section-title">📋 Tableau de Synthèse des Profils</div>',
                unsafe_allow_html=True)
    cluster_profiles = df_assigned.groupby("CLUSTER").agg(
        Effectif=("N", "count"),
        IMC_moyen=("IMC", "mean"),
        Duree_moy=("DUREE_MIN", "mean"),
        NbSeries_moy=("NB_SERIES", "mean"),
        PDS_moyen=("PDS_TOTAL", "mean"),
        Kerma_moyen=("KERMA_mGy", "mean"),
        DoseEff_moy=("DOSE_EFF", "mean"),
    ).round(1).reset_index()
    cluster_profiles["Profil Clinique"] = cluster_profiles["CLUSTER"].map(CLUSTER_NAMES)
    cluster_profiles["Recommandation"] = cluster_profiles["CLUSTER"].map(CLUSTER_RECO)
    cluster_profiles = cluster_profiles.drop(columns=["CLUSTER"])
    cols_order = ["Profil Clinique", "Effectif", "PDS_moyen", "Kerma_moyen",
                  "Duree_moy", "IMC_moyen", "NbSeries_moy", "DoseEff_moy", "Recommandation"]
    st.dataframe(cluster_profiles[cols_order].rename(columns={
        "PDS_moyen": "PDS moy (Gy·cm²)", "Kerma_moyen": "Kérma moy (mGy)",
        "Duree_moy": "Durée moy (min)", "IMC_moyen": "IMC moy",
        "NbSeries_moy": "Séries moy", "DoseEff_moy": "Dose Eff moy (mSv)",
    }), use_container_width=True)

    # ── Détail technique (expander) ───────────────────────────────────────────
    with st.expander("🔬 Détail technique — Méthode de classification (pour les experts)"):
        st.markdown("""
        <div class="explain-box">
          <h4>Algorithme utilisé</h4>
          <div class="explain-item">
            <b>Méthode :</b> K-Means (k=4) — algorithme de regroupement non supervisé basé sur la distance euclidienne
            dans l'espace des variables dosimétriques normalisées (StandardScaler).
          </div>
          <div class="explain-item">
            <b>Variables d'entrée :</b> IMC, Durée de scopie, Nb séries, Nb images, PDS Total, Kérma air.
          </div>
          <div class="explain-item">
            <b>Réduction de dimension :</b> ACP (Analyse en Composantes Principales) — 3 composantes.
            Les axes des graphiques correspondent aux 3 premières composantes principales.
          </div>
        </div>
        """, unsafe_allow_html=True)

        best_sil = round(max(sil_scores), 3)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clusters K-Means", 4)
        m2.metric("Score Silhouette optimal", best_sil, help="Plus proche de 1 = meilleure séparation")
        m3.metric("Variance expliquée (PC1+PC2)", f"{(pca_var_exp[0]+pca_var_exp[1])*100:.1f}%")
        m4.metric("Variance expliquée (PC1+PC2+PC3)", f"{sum(pca_var_exp)*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        k_range = list(range(2, 2 + len(elbow_inertias)))
        el1, el2 = st.columns(2)
        with el1:
            fig_elbow = go.Figure(go.Scatter(
                x=k_range, y=elbow_inertias, mode="lines+markers",
                marker=dict(color="#22d3ee", size=8), line=dict(color="#22d3ee"),
                name="Inertie",
            ))
            fig_elbow.add_vline(x=4, line_dash="dash", line_color="#f87171",
                                annotation_text="K=4 retenu", annotation_font_color="#f87171")
            fig_elbow.update_layout(
                template="plotly_dark", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
                title="Méthode Elbow — Choix du nombre de groupes",
                xaxis_title="Nombre de groupes (K)", yaxis_title="Inertie intra-cluster",
                height=300, margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
        with el2:
            sil_colors = ["#f87171" if k == 4 else "#22d3ee" for k in k_range]
            fig_sil = go.Figure(go.Bar(
                x=k_range, y=sil_scores, marker_color=sil_colors,
                text=[f"{s:.3f}" for s in sil_scores], textposition="outside",
            ))
            fig_sil.add_annotation(
                x=4, y=max(sil_scores), text="★ K=4 sélectionné",
                showarrow=True, arrowcolor="#f87171", font=dict(color="#f87171", size=11),
            )
            fig_sil.update_layout(
                template="plotly_dark", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
                title="Score Silhouette — Qualité de la séparation par K",
                xaxis_title="Nombre de groupes (K)", yaxis_title="Score Silhouette",
                height=300, margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_sil, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 3 — ATYPIQUES & RISQUES
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    main_df = st.session_state.main_dataset
    df_assigned = assign_cluster_anomaly(
        main_df, cl_scaler, km_model, iso_model, pca_model, CLUSTER_REMAP)

    st.markdown('<div class="section-title">⚠️ Détection des Patients Atypiques (IsolationForest)</div>',
                unsafe_allow_html=True)

    n_atypiques = (df_assigned["ANOMALIE"] == -1).sum()
    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("Patients atypiques", n_atypiques,
               delta=f"{n_atypiques/len(df_assigned)*100:.1f}%", delta_color="inverse")
    ac2.metric("Patients normaux", len(df_assigned) - n_atypiques)
    ac3.metric("Kérma > 2000 mGy", int((main_df["KERMA_mGy"] > 2000).sum()),
               help="Suivi dermatologique requis")
    ac4.metric("PDS > NRD France", int(df_assigned.apply(
        lambda r: r["PDS_TOTAL"] > (45 if r["PROC_TYPE"] == "CORO" else 85), axis=1).sum()))

    # Alerte si le dernier patient simulé est atypique
    last_sim = df_assigned[df_assigned["SOURCE"] == "Simulé"]
    if len(last_sim) > 0 and last_sim.iloc[-1]["ANOMALIE"] == -1:
        st.markdown(
            '<div class="alert-critique">⚠️ <b>Le dernier patient simulé est ATYPIQUE</b> — '
            'Profil dosimétrique inhabituellement élevé. Vérification recommandée.</div>',
            unsafe_allow_html=True)

    df_atyp = df_assigned[df_assigned["ANOMALIE"] == -1].copy()
    df_norm = df_assigned[df_assigned["ANOMALIE"] ==  1].copy()

    # Scatter atypiques
    fig_at = go.Figure()
    fig_at.add_trace(go.Scatter(
        x=df_norm["DUREE_MIN"], y=df_norm["PDS_TOTAL"], mode="markers", name="Normal",
        marker=dict(color="#4ade80", size=6, opacity=0.5,
                    symbol=["diamond" if s == "Simulé" else "circle" for s in df_norm["SOURCE"]]),
    ))
    fig_at.add_trace(go.Scatter(
        x=df_atyp["DUREE_MIN"], y=df_atyp["PDS_TOTAL"], mode="markers", name="Atypique ⚠️",
        marker=dict(color="#f87171", size=12, opacity=0.9,
                    symbol=["diamond" if s == "Simulé" else "x" for s in df_atyp["SOURCE"]],
                    line=dict(width=2, color="#f87171")),
    ))
    fig_at.update_layout(
        template="plotly_dark", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
        title="Patients atypiques — Durée vs PDS (◆ = simulé)",
        xaxis_title="Durée scopie (min)", yaxis_title="PDS (Gy·cm²)",
        height=400, margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_at, use_container_width=True)

    # Tableau des atypiques
    st.markdown("**Liste des patients atypiques à surveiller**")
    atyp_cols = ["N", "SOURCE", "SEXE", "AGE", "IMC", "PROC_TYPE",
                 "DUREE_MIN", "NB_IMAGES", "NB_SERIES", "PDS_TOTAL", "KERMA_mGy", "CLASSE"]
    st.dataframe(df_atyp[atyp_cols].sort_values("PDS_TOTAL", ascending=False).head(20),
                 use_container_width=True)
    st.download_button(
        "⬇️ Exporter patients atypiques (CSV)",
        df_atyp[atyp_cols].to_csv(index=False).encode("utf-8"),
        "patients_atypiques.csv", "text/csv",
    )

    # Comparaison atypiques vs normaux
    st.markdown('<div class="section-title">📊 Comparaison Atypiques vs Normaux</div>',
                unsafe_allow_html=True)
    comp_cols = ["PDS_TOTAL", "KERMA_mGy", "DUREE_MIN", "NB_IMAGES", "NB_SERIES", "IMC"]
    comp_data = []
    for c in comp_cols:
        moy_at = df_atyp[c].mean() if len(df_atyp) else 0
        moy_no = df_norm[c].mean() if len(df_norm) else 1
        comp_data.append({
            "Variable":          c,
            "Atypiques (moy)":   round(moy_at, 2),
            "Normaux (moy)":     round(moy_no, 2),
            "Ratio (At/No)":     round(moy_at / max(moy_no, 0.001), 2),
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    # Boxplots Kérma par classe
    st.markdown('<div class="section-title">📦 Boxplots — Kérma & PDS par classe de risque</div>',
                unsafe_allow_html=True)
    bx1, bx2 = st.columns(2)
    with bx1:
        fig_bk = px.box(main_df, x="CLASSE", y="KERMA_mGy", color="CLASSE",
                        points="all",
                        color_discrete_map={c: CLINICAL_RISK[c]["color"] for c in CLINICAL_RISK},
                        title="Kérma (mGy) par classe de risque ICRP",
                        template="plotly_dark",
                        category_orders={"CLASSE": ["A", "B", "C", "D"]})
        fig_bk.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#0a1628", height=360)
        st.plotly_chart(fig_bk, use_container_width=True)
    with bx2:
        fig_bp = px.box(main_df, x="CLASSE", y="PDS_TOTAL", color="CLASSE",
                        points="all",
                        color_discrete_map={c: CLINICAL_RISK[c]["color"] for c in CLINICAL_RISK},
                        title="PDS Total (Gy·cm²) par classe de risque ICRP",
                        template="plotly_dark",
                        category_orders={"CLASSE": ["A", "B", "C", "D"]})
        fig_bp.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#0a1628", height=360)
        st.plotly_chart(fig_bp, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # NOUVELLES SECTIONS — A. Score d'anomalie Isolation Forest
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🔢 A. Score d\'Anomalie Isolation Forest (Brut)</div>',
                unsafe_allow_html=True)

    score_min = df_assigned["SCORE_ANOMALIE"].min()
    score_max = df_assigned["SCORE_ANOMALIE"].max()
    score_moy = df_assigned["SCORE_ANOMALIE"].mean()
    threshold_score = df_assigned.loc[df_assigned["ANOMALIE"] == -1, "SCORE_ANOMALIE"].max() \
        if len(df_atyp) > 0 else score_min

    sa1, sa2, sa3 = st.columns(3)
    sa1.markdown(f"""<div class="kpi-card" style="--kpi-color:#f87171">
      <div class="kpi-value" style="color:#f87171">{score_min:.4f}</div>
      <div class="kpi-label">Score minimum</div>
      <div class="kpi-sub">Patient le plus atypique</div>
    </div>""", unsafe_allow_html=True)
    sa2.markdown(f"""<div class="kpi-card" style="--kpi-color:#4ade80">
      <div class="kpi-value" style="color:#4ade80">{score_max:.4f}</div>
      <div class="kpi-label">Score maximum</div>
      <div class="kpi-sub">Patient le plus normal</div>
    </div>""", unsafe_allow_html=True)
    sa3.markdown(f"""<div class="kpi-card" style="--kpi-color:#22d3ee">
      <div class="kpi-value" style="color:#22d3ee">{score_moy:.4f}</div>
      <div class="kpi-label">Score moyen</div>
      <div class="kpi-sub">Cohorte complète</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tableau avec score d'anomalie trié décroissant (plus atypique en premier)
    score_cols = ["N", "SOURCE", "SEXE", "AGE", "PROC_TYPE", "PDS_TOTAL",
                  "KERMA_mGy", "IMC", "CLASSE", "SCORE_ANOMALIE"]
    df_score_display = df_assigned[score_cols].copy()
    df_score_display["SCORE_ANOMALIE"] = df_score_display["SCORE_ANOMALIE"].round(4)
    df_score_display = df_score_display.sort_values("SCORE_ANOMALIE", ascending=True)
    st.dataframe(df_score_display.head(30), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # B. Histogramme des scores d'anomalie
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">📊 B. Histogramme des Scores d\'Anomalie</div>',
                unsafe_allow_html=True)

    pct_atypiques = n_atypiques / len(df_assigned) * 100

    fig_hist_score = go.Figure()
    fig_hist_score.add_trace(go.Histogram(
        x=df_norm["SCORE_ANOMALIE"],
        nbinsx=30,
        name="Normal",
        marker_color="#4ade80",
        opacity=0.8,
    ))
    fig_hist_score.add_trace(go.Histogram(
        x=df_atyp["SCORE_ANOMALIE"],
        nbinsx=15,
        name="Atypique",
        marker_color="#f87171",
        opacity=0.9,
    ))
    fig_hist_score.add_vline(
        x=threshold_score,
        line_dash="dash", line_color="#fbbf24", line_width=2,
        annotation_text=f"Seuil IsolationForest ({pct_atypiques:.1f}% atypiques)",
        annotation_font_color="#fbbf24",
        annotation_position="top left",
    )
    fig_hist_score.update_layout(
        template="plotly_dark", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
        title=f"Distribution des scores d'anomalie — {pct_atypiques:.1f}% de patients atypiques",
        xaxis_title="Score d'anomalie (plus bas = plus atypique)",
        yaxis_title="Effectif",
        barmode="overlay",
        height=380, margin=dict(t=50, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_hist_score, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # C. Carte PCA améliorée — anomalies
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🗺️ C. Carte PCA 2D — Anomalies</div>',
                unsafe_allow_html=True)

    _pca2_norm = df_assigned[df_assigned["ANOMALIE"] == 1].copy()
    _pca2_atyp = df_assigned[df_assigned["ANOMALIE"] == -1].copy()

    fig_pca_an = go.Figure()
    if len(_pca2_norm) > 0:
        fig_pca_an.add_trace(go.Scatter(
            x=_pca2_norm["PC1"], y=_pca2_norm["PC2"],
            mode="markers",
            name="Normal",
            marker=dict(color="#4ade80", size=7, opacity=0.55),
            customdata=np.stack([
                _pca2_norm.get("N", pd.Series(["?"] * len(_pca2_norm))).values,
                _pca2_norm["PDS_TOTAL"].values,
                _pca2_norm["KERMA_mGy"].values,
                _pca2_norm["IMC"].values,
                _pca2_norm["CLASSE"].values,
                _pca2_norm["PROC_TYPE"].values,
            ], axis=-1),
            hovertemplate=(
                "<b>N : %{customdata[0]}</b><br>"
                "PDS : %{customdata[1]:.1f} Gy·cm²<br>"
                "Kérma : %{customdata[2]:.0f} mGy<br>"
                "IMC : %{customdata[3]:.1f}<br>"
                "Classe : %{customdata[4]}<br>"
                "Procédure : %{customdata[5]}<extra></extra>"
            ),
        ))
    if len(_pca2_atyp) > 0:
        fig_pca_an.add_trace(go.Scatter(
            x=_pca2_atyp["PC1"], y=_pca2_atyp["PC2"],
            mode="markers",
            name="⚠️ Atypique",
            marker=dict(color="#f87171", size=14, opacity=0.95,
                        symbol="x", line=dict(width=2.5, color="#f87171")),
            customdata=np.stack([
                _pca2_atyp.get("N", pd.Series(["?"] * len(_pca2_atyp))).values,
                _pca2_atyp["PDS_TOTAL"].values,
                _pca2_atyp["KERMA_mGy"].values,
                _pca2_atyp["IMC"].values,
                _pca2_atyp["CLASSE"].values,
                _pca2_atyp["PROC_TYPE"].values,
            ], axis=-1),
            hovertemplate=(
                "<b>⚠️ ATYPIQUE N : %{customdata[0]}</b><br>"
                "PDS : %{customdata[1]:.1f} Gy·cm²<br>"
                "Kérma : %{customdata[2]:.0f} mGy<br>"
                "IMC : %{customdata[3]:.1f}<br>"
                "Classe : %{customdata[4]}<br>"
                "Procédure : %{customdata[5]}<extra></extra>"
            ),
        ))
    fig_pca_an.update_layout(
        template="plotly_dark", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
        title="PCA 2D — Détection des patients atypiques",
        xaxis_title=f"PC1 ({pca_var_exp[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca_var_exp[1]*100:.1f}%)",
        height=430, margin=dict(t=50, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_pca_an, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # D. Analyse automatique du dernier patient simulé
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🔬 D. Analyse du Dernier Patient Simulé</div>',
                unsafe_allow_html=True)

    last_sims_all = df_assigned[df_assigned["SOURCE"] == "Simulé"]
    if len(last_sims_all) > 0:
        lp = last_sims_all.iloc[-1]
        lp_cluster      = int(lp["CLUSTER"]) if not pd.isna(lp["CLUSTER"]) else 0
        lp_classe       = lp.get("CLASSE", "A")
        lp_score_an     = lp["SCORE_ANOMALIE"]
        lp_is_atyp      = lp["ANOMALIE"] == -1
        lp_pds          = lp["PDS_TOTAL"]
        lp_kerma        = lp["KERMA_mGy"]
        lp_pct_pds      = (df_assigned["PDS_TOTAL"] < lp_pds).mean() * 100
        lp_pct_kerma    = (df_assigned["KERMA_mGy"] < lp_kerma).mean() * 100
        lp_nrd_pds      = 45 if lp["PROC_TYPE"] == "CORO" else 85
        lp_conforme_nrd = lp_pds <= lp_nrd_pds
        lp_percentile_cohort = lp_pct_pds

        # Texte d'interprétation automatique
        if lp_pct_pds > 75:
            interp_pds = f"une exposition supérieure à {lp_pct_pds:.0f}% des patients de la cohorte"
        elif lp_pct_pds < 25:
            interp_pds = f"une exposition inférieure à {100-lp_pct_pds:.0f}% des patients de la cohorte"
        else:
            interp_pds = f"une exposition dans la moyenne de la cohorte ({lp_pct_pds:.0f}e percentile)"

        interp_atyp = "Un profil dosimétrique atypique a été détecté — vérification recommandée." \
            if lp_is_atyp else "Aucun signe d'atypie majeure n'a été détecté."
        interp_nrd  = "conforme aux niveaux de référence diagnostiques (NRD)" \
            if lp_conforme_nrd else "dépassant les niveaux de référence diagnostiques (NRD)"
        interp_full = (
            f"Ce patient présente {interp_pds}. "
            f"Son exposition est {interp_nrd}. "
            f"{interp_atyp}"
        )

        lp_color = "#f87171" if lp_is_atyp else "#4ade80"
        lp_status_html = (
            '<span style="background:#450a0a;color:#f87171;border:1px solid #991b1b;'
            'border-radius:6px;padding:2px 10px;font-size:11px;font-weight:700;">⚠️ ATYPIQUE</span>'
            if lp_is_atyp else
            '<span style="background:#052e16;color:#4ade80;border:1px solid #166534;'
            'border-radius:6px;padding:2px 10px;font-size:11px;font-weight:700;">✅ NORMAL</span>'
        )
        lp_nrd_html = (
            '<span style="color:#4ade80;font-weight:700;">✅ Conforme NRD</span>'
            if lp_conforme_nrd else
            '<span style="color:#f87171;font-weight:700;">❌ Dépasse NRD</span>'
        )
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0a1628,#0d2137);
                    border:2px solid {lp_color}40;border-radius:16px;padding:22px 26px;margin-bottom:12px;">
          <div style="font-size:14px;font-weight:800;color:#22d3ee;margin-bottom:16px;">
            🧬 Analyse du dernier patient simulé
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:16px;">
            <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:12px;text-align:center;">
              <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.08em">Cluster</div>
              <div style="font-size:18px;font-weight:800;color:{CLUSTER_COLORS.get(lp_cluster,"#94a3b8")}">
                {CLUSTER_NAMES.get(lp_cluster,"—")}
              </div>
            </div>
            <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:12px;text-align:center;">
              <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.08em">Classe clinique</div>
              <div style="font-size:18px;font-weight:800;color:{CLINICAL_RISK.get(lp_classe,{}).get("color","#94a3b8")}">
                {CLINICAL_RISK.get(lp_classe,{}).get("label","—")}
              </div>
            </div>
            <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:12px;text-align:center;">
              <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.08em">Score anomalie</div>
              <div style="font-size:18px;font-weight:800;color:{lp_color}">{lp_score_an:.4f}</div>
            </div>
            <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:12px;text-align:center;">
              <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.08em">Percentile PDS</div>
              <div style="font-size:18px;font-weight:800;color:#a78bfa">{lp_pct_pds:.0f}e</div>
            </div>
            <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:12px;text-align:center;">
              <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.08em">Percentile Kérma</div>
              <div style="font-size:18px;font-weight:800;color:#f59e0b">{lp_pct_kerma:.0f}e</div>
            </div>
            <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:12px;text-align:center;">
              <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.08em">Statut</div>
              <div style="margin-top:4px">{lp_status_html}</div>
            </div>
          </div>
          <div style="background:#080f1a;border:1px solid #1e3a5f;border-radius:10px;padding:14px;margin-bottom:12px;">
            <div style="font-size:10px;color:#22d3ee;text-transform:uppercase;letter-spacing:.1em;
                        font-weight:700;margin-bottom:8px;">📋 Interprétation médicale automatique</div>
            <div style="font-size:13px;color:#cbd5e1;line-height:1.7;font-style:italic;">
              "{interp_full}"
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:10px;font-size:13px;">
            <span style="color:#64748b">Conformité NRD :</span> {lp_nrd_html}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-info">ℹ️ Aucun patient simulé disponible. '
                    'Effectuez d\'abord une prédiction dans l\'onglet Accueil.</div>',
                    unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # E. Bloc d'alerte dosimétrique intelligent
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🚨 E. Alertes Dosimétriques Intelligentes</div>',
                unsafe_allow_html=True)

    if st.session_state.pred_pds is not None:
        _pds_al    = st.session_state.pred_pds
        _kerma_al  = st.session_state.pred_kerma
        _cls_al    = st.session_state.classe
        _atyp_al   = st.session_state.get("is_atypique", False)
        _proc_al   = st.session_state.patient_data["proc_type"]
        _nrd_al    = 45 if _proc_al == "CORO" else 85

        _triggers = []
        if _kerma_al > 2000:
            _triggers.append(("Kérma > 2000 mGy", f"Kérma = {_kerma_al:.0f} mGy",
                               "Suivi dermatologique obligatoire (J10–J14). Effet déterministe possible.",
                               "☢️"))
        if _pds_al > _nrd_al:
            _triggers.append(("PDS dépasse le NRD", f"PDS = {_pds_al:.1f} Gy·cm² (NRD = {_nrd_al})",
                               "Audit dosimétrique recommandé. Revoir le protocole de la procédure.",
                               "📊"))
        if _atyp_al:
            _triggers.append(("Profil atypique détecté", "IsolationForest : profil inhabituel",
                               "Vérifier la cohérence des paramètres. Comparer avec des cas similaires.",
                               "⚠️"))

        if _triggers:
            _niveau = "CRITIQUE" if len(_triggers) >= 2 or _kerma_al > 2000 else "ÉLEVÉ"
            _niveau_color = "#f87171" if _niveau == "CRITIQUE" else "#fb923c"
            _causes_html = "".join([
                f'<div class="alert-premium-row">'
                f'<span class="alert-premium-icon">{icon}</span>'
                f'<div><div class="alert-premium-label">{cause}</div>'
                f'<div class="alert-premium-text">{detail}</div>'
                f'<div class="alert-premium-text" style="color:#fca5a5;margin-top:3px">💡 {reco}</div></div>'
                f'</div>'
                for cause, detail, reco, icon in _triggers
            ])
            st.markdown(f"""
            <div class="alert-premium-red">
              <div class="alert-title">
                🚨 ALERTE DOSIMÉTRIQUE — NIVEAU {_niveau}
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">
                <div style="background:#5c0f0f;border-radius:8px;padding:10px;text-align:center;">
                  <div style="font-size:10px;color:#dc2626;text-transform:uppercase;letter-spacing:.08em">Niveau d'alerte</div>
                  <div style="font-size:20px;font-weight:800;color:{_niveau_color}">{_niveau}</div>
                </div>
                <div style="background:#5c0f0f;border-radius:8px;padding:10px;text-align:center;">
                  <div style="font-size:10px;color:#dc2626;text-transform:uppercase;letter-spacing:.08em">Déclencheurs</div>
                  <div style="font-size:20px;font-weight:800;color:#f87171">{len(_triggers)}</div>
                </div>
              </div>
              {_causes_html}
              <div style="margin-top:14px;padding-top:12px;border-top:1px solid #7f1d1d;">
                <div style="font-size:11px;color:#dc2626;text-transform:uppercase;letter-spacing:.08em;font-weight:700;
                            margin-bottom:6px;">⚕️ Risque clinique global</div>
                <div style="font-size:13px;color:#fca5a5;line-height:1.6;">
                  {'Risque élevé à effet déterministe (érythème cutané). Suivi multidisciplinaire obligatoire.' 
                   if _kerma_al > 2000 else 
                   'Risque stochastique accru. Documenter et notifier au service de radioprotection.'}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#052e16;border:2px solid #166534;border-radius:14px;
                        padding:20px 24px;text-align:center;">
              <div style="font-size:22px;margin-bottom:8px;">✅</div>
              <div style="font-size:16px;font-weight:700;color:#4ade80">Aucune alerte dosimétrique</div>
              <div style="font-size:13px;color:#86efac;margin-top:6px">
                Tous les indicateurs sont dans les limites acceptables pour ce patient.
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-info">ℹ️ Prédisez d\'abord un patient (onglet Accueil) '
                    'pour activer les alertes intelligentes.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 4 — CONFORMITÉ NRD
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    main_df = st.session_state.main_dataset
    st.markdown('<div class="section-title">🌍 Comparaison aux NRD Internationaux</div>',
                unsafe_allow_html=True)

    if st.session_state.pred_pds is None:
        st.markdown('<div class="alert-info">ℹ️ Prédisez d\'abord un patient (onglet Accueil) '
                    'pour une comparaison personnalisée. En attendant, la médiane de la cohorte '
                    'est utilisée.</div>', unsafe_allow_html=True)
        pds_cmp   = main_df["PDS_TOTAL"].median()
        proc_cmp  = "CORO"
        kerma_cmp = main_df["KERMA_mGy"].median()
    else:
        pds_cmp   = st.session_state.pred_pds
        kerma_cmp = st.session_state.pred_kerma
        proc_cmp  = st.session_state.patient_data["proc_type"]

    # Jauges NRD
    nrd_cols = st.columns(3)
    for idx, (org, vals) in enumerate(NRD.items()):
        nrd_pds_v = vals[f"PDS_{proc_cmp}"]
        pct = pds_cmp / nrd_pds_v * 100
        color = "#4ade80" if pct <= 80 else "#fbbf24" if pct <= 100 else "#f87171"
        fig_jg = go.Figure(go.Indicator(
            mode="gauge+number", value=pct,
            title={"text": f"{org} — {proc_cmp}"},
            gauge={
                "axis": {"range": [0, 200]}, "bar": {"color": color},
                "steps": [
                    {"range": [0, 80],   "color": "#052e16"},
                    {"range": [80, 100], "color": "#422006"},
                    {"range": [100, 200],"color": "#450a0a"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 100},
            },
            number={"suffix": "%", "valueformat": ".0f"},
        ))
        fig_jg.update_layout(paper_bgcolor="#0a1628", font_color="#94a3b8",
                             height=220, margin=dict(t=50, b=10, l=10, r=10))
        nrd_cols[idx % 3].plotly_chart(fig_jg, use_container_width=True)
        nrd_cols[idx % 3].markdown(
            f"<center style='font-size:11px;color:#64748b'>NRD = {nrd_pds_v} Gy·cm² | "
            f"{'✅ Conforme' if pct <= 100 else '❌ Dépassé'}</center>",
            unsafe_allow_html=True,
        )

    # Tableau comparatif
    st.markdown('<div class="section-title">📋 Tableau Comparatif</div>', unsafe_allow_html=True)
    comp_rows = []
    for org, vals in NRD.items():
        nrd_pds_v   = vals[f"PDS_{proc_cmp}"]
        nrd_kerma_v = vals[f"Kerma_{proc_cmp}"]
        comp_rows.append({
            "Organisme":          org,
            "NRD PDS (Gy·cm²)":  nrd_pds_v,
            "PDS Patient":        round(pds_cmp, 1),
            "% NRD PDS":          f"{pds_cmp/nrd_pds_v*100:.0f}%",
            "NRD Kérma (mGy)":   nrd_kerma_v,
            "Kérma Patient":      round(kerma_cmp, 0),
            "% NRD Kérma":        f"{kerma_cmp/nrd_kerma_v*100:.0f}%",
            "Statut PDS":         "✅" if pds_cmp <= nrd_pds_v else "❌",
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

    # Taux de dépassement sur la cohorte
    st.markdown('<div class="section-title">📊 Taux de dépassement NRD — cohorte complète</div>',
                unsafe_allow_html=True)
    dep_rows = []
    for org, vals in NRD.items():
        for pt in ["CORO", "ANGIO"]:
            sub = main_df[main_df["PROC_TYPE"] == pt]
            if len(sub) == 0:
                continue
            nrd_v = vals[f"PDS_{pt}"]
            dep   = (sub["PDS_TOTAL"] > nrd_v).mean() * 100
            dep_rows.append({
                "Organisme": org, "Type": pt,
                "NRD PDS": nrd_v, "% Dépassement": round(dep, 1)
            })
    dep_df = pd.DataFrame(dep_rows)
    fig_dep = px.bar(
        dep_df, x="Organisme", y="% Dépassement", color="Type", barmode="group",
        title="Taux de dépassement NRD par référentiel (cohorte réels + simulés)",
        template="plotly_dark",
        color_discrete_map={"CORO": "#22d3ee", "ANGIO": "#a78bfa"},
    )
    fig_dep.add_hline(y=50, line_dash="dash", line_color="#f87171",
                      annotation_text="Seuil d'alerte 50%", annotation_font_color="#f87171")
    fig_dep.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
                          height=380, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_dep, use_container_width=True)

    # Distribution PDS par rapport au NRD de chaque organisme
    with st.expander("📈 Distribution PDS vs NRD — Vue détaillée par organisme"):
        sel_org = st.selectbox("Organisme", list(NRD.keys()))
        for pt in ["CORO", "ANGIO"]:
            sub = main_df[main_df["PROC_TYPE"] == pt]["PDS_TOTAL"]
            if len(sub) == 0:
                continue
            nrd_v = NRD[sel_org][f"PDS_{pt}"]
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=sub, nbinsx=25,
                marker_color="#22d3ee" if pt == "CORO" else "#a78bfa",
                name=pt, opacity=0.75,
            ))
            fig_dist.add_vline(x=nrd_v, line_dash="dash", line_color="#f87171", line_width=2,
                               annotation_text=f"NRD {sel_org} {pt} = {nrd_v}",
                               annotation_font_color="#f87171")
            fig_dist.update_layout(
                template="plotly_dark", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
                title=f"Distribution PDS — {pt} | NRD {sel_org} = {nrd_v} Gy·cm²",
                xaxis_title="PDS Total (Gy·cm²)", yaxis_title="Effectif",
                height=280, margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
