

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import joblib
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Configure matplotlib dark mode aesthetics
plt.rcParams.update({
    "figure.facecolor": "#00000000",
    "axes.facecolor": "#00000000",
    "text.color": "#f8fafc",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "axes.edgecolor": "#475569",
    "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans"],
})


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="SegmentIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HELPERS
# ============================================================

def html(content):
    """Render HTML without indentation turning it into a code block."""
    st.markdown(textwrap.dedent(content).strip(), unsafe_allow_html=True)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        box-sizing: border-box;
    }

    /* App Background with ambient glow */
    .stApp {
        background-color: #080c14 !important;
        background-image: 
            radial-gradient(at 10% 12%, rgba(99, 102, 241, 0.12) 0px, transparent 45%),
            radial-gradient(at 88% 18%, rgba(168, 85, 247, 0.09) 0px, transparent 45%),
            radial-gradient(at 50% 80%, rgba(16, 185, 129, 0.07) 0px, transparent 45%) !important;
        background-attachment: fixed !important;
        color: #f1f5f9 !important;
    }

    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: transparent !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: none !important;
        width: 100% !important;
        padding: 24px 32px 40px 32px !important;
    }

    section.main > div {
        max-width: none !important;
    }

    /* Element spacing */
    .element-container {
        margin-bottom: 0 !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.75rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(7, 10, 18, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    section[data-testid="stSidebar"] > div {
        padding: 20px 16px 24px 16px;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .brand {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff 40%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 3px;
    }

    .brand-sub {
        color: #64748b !important;
        font-size: 11px;
        margin-bottom: 22px;
        font-weight: 500;
        letter-spacing: 0.2px;
    }

    .side-title {
        color: #475569 !important;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin: 22px 0 8px 4px;
        text-transform: uppercase;
    }

    /* Sidebar Buttons */
    section[data-testid="stSidebar"] button {
        min-height: 42px !important;
        border-radius: 12px !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: #94a3b8 !important;
        transition: all .2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: left !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 8px 14px !important;
    }

    section[data-testid="stSidebar"] button:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        transform: translateX(3px);
    }

    section[data-testid="stSidebar"] button[kind="primary"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.45) !important;
    }

    /* Topbar */
    .topbar {
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
    }

    .search {
        width: 390px;
        height: 38px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        background: rgba(18, 24, 38, 0.6);
        backdrop-filter: blur(12px);
        display: flex;
        align-items: center;
        padding: 0 16px;
        color: #94a3b8;
        font-size: 12px;
    }

    .top-user {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #e2e8f0;
        font-size: 12px;
        font-weight: 700;
    }

    .avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        font-weight: 800;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
    }

    .top-icon {
        position: relative;
        text-align: center;
        font-size: 19px;
        padding-top: 6px;
        color: #94a3b8;
        transition: all .2s ease;
    }
    .top-icon:hover { color: #f8fafc; }

    .notify-dot {
        position: absolute;
        top: 2px;
        right: 8px;
        color: #f43f5e;
        font-size: 14px;
        filter: drop-shadow(0 0 6px rgba(244, 63, 94, 0.8));
    }

    /* Hero Banner */
    .hero {
        min-height: 135px;
        border-radius: 18px;
        padding: 26px 34px;
        color: white;
        overflow: hidden;
        position: relative;
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.85) 0%, rgba(15, 23, 42, 0.95) 50%, rgba(17, 24, 39, 0.9) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(16px);
    }

    .hero:before {
        content: "";
        position: absolute;
        top: -50px;
        right: 15%;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero:after {
        content: "";
        position: absolute;
        right: 5%;
        bottom: -80px;
        width: 500px;
        height: 160px;
        border-radius: 50%;
        border-top: 2px solid rgba(129, 140, 248, 0.2);
        transform: rotate(-6deg);
        pointer-events: none;
    }

    .hero-greeting {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 700;
        color: #a5b4fc;
        background: rgba(99, 102, 241, 0.16);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 8px;
        letter-spacing: 0.3px;
    }

    .hero-title {
        font-size: 28px;
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff 60%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-desc {
        font-size: 13px;
        line-height: 1.55;
        color: #94a3b8;
        max-width: 740px;
    }

    .hero-art {
        position: absolute;
        right: 42px;
        top: 32px;
        font-size: 38px;
        letter-spacing: 10px;
        color: rgba(99, 102, 241, 0.3);
        font-weight: 300;
    }

    /* KPI Cards with Glowing Accents */
    .dash-kpi, .kpi {
        border-radius: 16px !important;
        padding: 16px 20px !important;
        min-height: 114px !important;
        background: rgba(18, 24, 38, 0.72) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25) !important;
        position: relative;
        overflow: hidden;
        transition: all .25s ease;
    }

    .dash-kpi:hover, .kpi:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4) !important;
    }

    .blue-kpi, .kpi.blue {
        border-left: 3px solid #3b82f6 !important;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(18, 24, 38, 0.75) 100%) !important;
    }

    .purple-kpi, .kpi.purple {
        border-left: 3px solid #a855f7 !important;
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.12) 0%, rgba(18, 24, 38, 0.75) 100%) !important;
    }

    .green-kpi, .kpi.green {
        border-left: 3px solid #10b981 !important;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(18, 24, 38, 0.75) 100%) !important;
    }

    .pink-kpi, .kpi.pink {
        border-left: 3px solid #f43f5e !important;
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.12) 0%, rgba(18, 24, 38, 0.75) 100%) !important;
    }

    .kpi-icon {
        width: 38px;
        height: 38px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
        margin-bottom: 6px;
    }

    .blue-kpi .kpi-icon { background: rgba(59, 130, 246, 0.2); color: #60a5fa; box-shadow: 0 0 15px rgba(59, 130, 246, 0.35); }
    .purple-kpi .kpi-icon { background: rgba(168, 85, 247, 0.2); color: #c084fc; box-shadow: 0 0 15px rgba(168, 85, 247, 0.35); }
    .green-kpi .kpi-icon { background: rgba(16, 185, 129, 0.2); color: #34d399; box-shadow: 0 0 15px rgba(16, 185, 129, 0.35); }
    .pink-kpi .kpi-icon { background: rgba(244, 63, 94, 0.2); color: #fb7185; box-shadow: 0 0 15px rgba(244, 63, 94, 0.35); }

    .dash-kpi .kpi-label, .kpi-label {
        font-size: 11px !important;
        color: #94a3b8 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        position: absolute;
        left: 64px;
        top: 18px;
    }

    .dash-kpi .kpi-value, .kpi-value {
        color: #f8fafc !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin-left: 64px;
        margin-top: 2px;
    }

    .dash-kpi .kpi-link, .kpi-link {
        display: none !important;
    }

    .kpi-sub {
        color: #64748b !important;
        font-size: 11px !important;
        margin-top: 4px;
    }

    /* Glassmorphic Cards */
    .dashboard-card, .card {
        background: rgba(18, 24, 38, 0.72) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3) !important;
        height: 100%;
        transition: border-color .2s ease;
    }

    .dashboard-card:hover, .card:hover {
        border-color: rgba(99, 102, 241, 0.3) !important;
    }

    .dashboard-card-title, .card-title {
        color: #f1f5f9 !important;
        font-size: 14px !important;
        font-weight: 800 !important;
        letter-spacing: -0.2px;
        margin-bottom: 12px !important;
    }

    .segment-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12px !important;
        color: #94a3b8 !important;
        padding: 6px 4px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .segment-row strong { color: #f8fafc !important; }

    .segment-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #818cf8;
        margin-right: 8px;
        box-shadow: 0 0 8px rgba(129, 140, 248, 0.6);
    }

    .chart-note {
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
    }

    .match-count {
        text-align: center;
        font-size: 11px;
        color: #a5b4fc;
        font-weight: 700;
        margin-top: 10px;
        padding: 9px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
    }

    .insight-mini, .mini {
        min-height: 80px;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        margin-bottom: 8px;
        background: rgba(25, 33, 50, 0.6) !important;
        backdrop-filter: blur(10px);
    }

    .insight-icon {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        float: left;
        margin-right: 10px;
        font-size: 11px;
        font-weight: 800;
    }

    .insight-icon.green { background: rgba(16, 185, 129, 0.15) !important; color: #34d399 !important; box-shadow: 0 0 10px rgba(16, 185, 129, 0.3); }
    .insight-icon.purple { background: rgba(168, 85, 247, 0.15) !important; color: #c084fc !important; box-shadow: 0 0 10px rgba(168, 85, 247, 0.3); }
    .insight-icon.teal { background: rgba(20, 184, 166, 0.15) !important; color: #2dd4bf !important; box-shadow: 0 0 10px rgba(20, 184, 166, 0.3); }

    .mini-label {
        color: #94a3b8 !important;
        font-size: 10px !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .mini-value {
        color: #f8fafc !important;
        font-size: 13px !important;
        font-weight: 700;
        margin-top: 3px;
    }

    /* Prediction Result Glow */
    .prediction-result, .result {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(6, 78, 59, 0.25)) !important;
        border: 1px solid rgba(16, 185, 129, 0.35) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        margin-top: 10px;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.14) !important;
    }

    .prediction-name, .result-name {
        font-size: 18px !important;
        font-weight: 800 !important;
        color: #34d399 !important;
        margin-top: 4px;
        letter-spacing: -0.3px;
    }

    .prediction-sub, .result-label {
        font-size: 11px !important;
        color: #a7f3d0 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .pill {
        display: inline-block;
        padding: 4px 12px;
        margin-top: 6px;
        border-radius: 20px;
        background: rgba(99, 102, 241, 0.2) !important;
        color: #a5b4fc !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        font-size: 11px !important;
        font-weight: 700;
    }

    /* Form Inputs */
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #cbd5e1 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    .stTextInput input, .stNumberInput input {
        height: 38px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        background: rgba(15, 23, 42, 0.85) !important;
        color: #f8fafc !important;
        font-size: 13px !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }

    div[data-baseweb="select"] > div {
        min-height: 38px !important;
        border-radius: 10px !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        background: rgba(15, 23, 42, 0.85) !important;
    }

    div[data-baseweb="select"] * {
        color: #f8fafc !important;
        font-size: 13px !important;
    }

    /* Buttons */
    .stButton > button {
        min-height: 38px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        color: white !important;
        background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
        transition: all .2s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca, #4f46e5) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(18, 24, 38, 0.72) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 11px !important;
        color: #94a3b8 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }

    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        color: #f8fafc !important;
        font-weight: 800 !important;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Profile Section */
    .profile-shell {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.9) 60%, rgba(17, 24, 39, 0.8) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 20px !important;
        padding: 28px !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(20px);
    }

    .profile-avatar-large {
        width: 86px;
        height: 86px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        color: #fff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 34px;
        font-weight: 800;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.5) !important;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    .profile-name-large {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin-top: 10px;
        letter-spacing: -0.5px;
    }

    .profile-role-large {
        font-size: 13px !important;
        color: #94a3b8 !important;
        margin-top: 4px;
    }

    .profile-badge {
        display: inline-block;
        margin-top: 12px;
        padding: 6px 14px;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.15) !important;
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.35) !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
    }

    .profile-info-card {
        background: rgba(18, 24, 38, 0.72) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 18px !important;
    }

    .profile-info-title {
        font-size: 11px !important;
        color: #64748b !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .profile-info-value {
        font-size: 22px !important;
        color: #f8fafc !important;
        font-weight: 800 !important;
        margin-top: 6px;
    }

    .profile-info-sub {
        font-size: 11px !important;
        color: #94a3b8 !important;
        margin-top: 4px;
    }

    .profile-action {
        background: rgba(18, 24, 38, 0.72) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 16px !important;
    }

    /* About Project Section */
    .about-hero {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        color: white;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 20px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4);
    }

    .about-hero-title {
        font-size: 28px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff 60%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .about-hero-text {
        font-size: 13px !important;
        line-height: 1.65;
        color: #94a3b8 !important;
        margin-top: 8px;
    }

    .about-card {
        background: rgba(18, 24, 38, 0.72) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    }

    .about-card-title {
        font-size: 16px !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        margin-bottom: 8px;
    }

    .about-card-text {
        font-size: 12px !important;
        color: #94a3b8 !important;
        line-height: 1.65;
    }

    .pipeline-step {
        background: rgba(25, 33, 50, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px;
        padding: 14px;
        text-align: center;
    }

    .pipeline-num {
        width: 28px;
        height: 28px;
        margin: 0 auto 6px;
        border-radius: 50%;
        background: #6366f1 !important;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.5);
        color: #fff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 800;
    }

    .pipeline-label {
        font-size: 12px !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }

    .tech-pill {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.15) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        color: #a5b4fc !important;
        font-size: 11px !important;
        font-weight: 700;
    }

    .section {
        font-size: 18px !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        margin: 16px 0 10px 1px;
        letter-spacing: -0.3px;
    }

    .page-title {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        margin: 0 !important;
    }

    .page-subtitle {
        font-size: 13px !important;
        color: #94a3b8 !important;
        margin-top: 4px !important;
        margin-bottom: 18px !important;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online { background: #10b981; box-shadow: 0 0 8px #10b981; }
    .status-offline { background: #f43f5e; box-shadow: 0 0 8px #f43f5e; }

    /* Hide Streamlit decoration */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "customer_segments.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "backend",
    "model",
    "customer_segmentation_model.pkl"
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")


# ============================================================
# LOAD
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model_data = load_model()
except Exception as e:
    st.error(f"Project files could not be loaded: {e}")
    st.stop()


feature_columns = model_data["feature_columns"]
preprocessor = model_data["preprocessor"]
cluster_names = model_data["cluster_names"]


# ============================================================
# API
# ============================================================

def api_online():
    try:
        response = requests.get(API_URL, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


API_ONLINE = api_online()


# ============================================================
# SESSION STATE
# ============================================================

if "recent_predictions" not in st.session_state:
    st.session_state.recent_predictions = []


# ============================================================
# DATA VALUES
# ============================================================

total_customers = len(df)

segment_counts = (
    df["Segment_Name"]
    .value_counts()
    .sort_index()
)

higher_count = int(
    df["Segment_Name"]
    .eq("Higher-Income Customers")
    .sum()
)

lower_count = int(
    df["Segment_Name"]
    .eq("Lower-Income Customers")
    .sum()
)

higher_pct = higher_count / total_customers * 100
lower_pct = lower_count / total_customers * 100


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html("""
    <div class="brand">◈ SegmentIQ</div>
    <div class="brand-sub">Intelligent Customer Analytics</div>
    <div class="side-title">MAIN MENU</div>
    """)

    nav_items = [
        ("🏠", "Dashboard"),
        ("👤", "My Profile"),
        ("🎯", "Customer Prediction"),
        ("📊", "Segment Analytics"),
        ("🔎", "Customer Explorer"),
        ("ℹ️", "About Project"),
    ]
    current_page = st.session_state.get("page_nav", "Dashboard")

    for icon, label in nav_items:
        active = current_page == label
        if st.button(
            f"{icon}  {label}",
            key=f"nav_{label.replace(' ', '_')}",
            use_container_width=True,
            type="primary" if active else "secondary"
        ):
            st.session_state.page_nav = label
            st.rerun()

    page = st.session_state.get("page_nav", "Dashboard")

    html('<div class="side-title">MODEL STATUS</div>')
    if API_ONLINE:
        html('<div style="color:#5ee2a8;font-size:12px;font-weight:700;">● FastAPI Connected</div>')
    else:
        html('<div style="color:#ff7f8d;font-size:12px;font-weight:700;">● FastAPI Offline</div>')

    html("""
    <div class="side-title">MODEL</div>
    <div style="font-size:11px;color:#93a4c6;">Algorithm</div>
    <div style="font-size:13px;font-weight:700;margin-bottom:10px;">K-Means</div>
    <div style="font-size:11px;color:#93a4c6;">Learning</div>
    <div style="font-size:13px;font-weight:700;margin-bottom:10px;">Unsupervised</div>
    <div style="font-size:11px;color:#93a4c6;">Clusters</div>
    <div style="font-size:13px;font-weight:700;">2</div>
    """)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    # ---------------- TOP BAR ----------------
    top_search, bell_col, profile_col = st.columns([8.6, 0.55, 1.65], gap="small")

    with top_search:
        dashboard_search = st.text_input(
            "Search",
            placeholder="Search customers, segments, or insights...",
            label_visibility="collapsed",
            key="dashboard_search"
        )

    with bell_col:
        st.markdown(
            "<div class='top-icon'>🔔<span class='notify-dot'>•</span></div>",
            unsafe_allow_html=True
        )

    with profile_col:
        profile_clicked = st.button(
            "👤  Pranav Rao",
            key="profile_button",
            use_container_width=True
        )

    if profile_clicked:
        st.session_state.page_nav = "My Profile"
        st.rerun()

    # ---------------- SEARCH ----------------
    if dashboard_search.strip():
        q = dashboard_search.strip().lower()
        mask = (
            df["Customer ID"].astype(str).str.lower().str.contains(q, na=False) |
            df["Segment_Name"].astype(str).str.lower().str.contains(q, na=False) |
            df["Gender"].astype(str).str.lower().str.contains(q, na=False) |
            df["Occupation"].astype(str).str.lower().str.contains(q, na=False)
        )
        results = df.loc[mask].head(8)
        if not results.empty:
            st.markdown("#### 🔎 Search Results")
            st.dataframe(
                results[["Customer ID", "Age", "Gender", "Occupation", "Income Level", "Segment_Name"]],
                hide_index=True,
                use_container_width=True,
                height=260
            )
        else:
            st.warning("No matching customers, segments, gender or occupation found.")

    # ---------------- HERO ----------------
    html(f"""
    <div class="hero dashboard-hero">
        <div class="hero-greeting">👋 Hello, Pranav!</div>
        <div class="hero-title">Welcome to Customer Intelligence Dashboard</div>
        <div class="hero-desc">
            Explore customer segments, understand behavioural patterns and gain valuable insights<br>
            from your data.
        </div>
        <div class="hero-art">◉ ◉ ◇</div>
    </div>
    """)

    # ---------------- KPI CARDS ----------------
    k1, k2, k3, k4 = st.columns(4, gap="small")

    with k1:
        html(f"""
        <div class="dash-kpi blue-kpi">
            <div class="kpi-icon">👥</div>
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
        </div>
        """)
        if st.button("View details →", key="kpi_total", use_container_width=True):
            st.session_state.page_nav = "Customer Explorer"
            st.rerun()

    with k2:
        html(f"""
        <div class="dash-kpi purple-kpi">
            <div class="kpi-icon">🧩</div>
            <div class="kpi-label">Customer Segments</div>
            <div class="kpi-value">{len(cluster_names)}</div>
        </div>
        """)
        if st.button("View details →", key="kpi_segments", use_container_width=True):
            st.session_state.page_nav = "Segment Analytics"
            st.rerun()

    with k3:
        html(f"""
        <div class="dash-kpi green-kpi">
            <div class="kpi-icon">↑</div>
            <div class="kpi-label">Higher-Income Customers</div>
            <div class="kpi-value">{higher_pct:.2f}%</div>
            <div class="kpi-link">{higher_count:,} customers</div>
        </div>
        """)
        if st.button("View details →", key="kpi_higher", use_container_width=True):
            st.session_state.page_nav = "Customer Explorer"
            st.session_state.explorer_segment = "Higher-Income Customers"
            st.rerun()

    with k4:
        html(f"""
        <div class="dash-kpi pink-kpi">
            <div class="kpi-icon">↓</div>
            <div class="kpi-label">Lower-Income Customers</div>
            <div class="kpi-value">{lower_pct:.2f}%</div>
            <div class="kpi-link">{lower_count:,} customers</div>
        </div>
        """)
        if st.button("View details →", key="kpi_lower", use_container_width=True):
            st.session_state.page_nav = "Customer Explorer"
            st.session_state.explorer_segment = "Lower-Income Customers"
            st.rerun()

    html('<div class="section dashboard-section">Customer Overview</div>')

    # ---------------- OVERVIEW ----------------
    overview_left, overview_mid, overview_right = st.columns([1.0, 1.55, .88], gap="small")

    with overview_left:
        html('<div class="dashboard-card"><div class="dashboard-card-title">🧩 Customer Segments</div>')
        fig, ax = plt.subplots(figsize=(3.1, 2.15), facecolor='none')
        ax.set_facecolor('none')
        vals = segment_counts.values
        donut_colors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]
        ax.pie(
            vals,
            startangle=90,
            autopct="%1.2f%%",
            pctdistance=.78,
            colors=donut_colors[:len(vals)],
            wedgeprops={"width": .38, "edgecolor": "#0d1322", "linewidth": 2},
            textprops={"fontsize": 8, "color": "#f1f5f9", "weight": "bold"}
        )
        ax.text(0, .05, f"{total_customers:,}", ha="center", va="center", fontsize=14, fontweight="bold", color="#ffffff")
        ax.text(0, -.15, "Total Customers", ha="center", va="center", fontsize=7, color="#94a3b8")
        ax.set(aspect="equal")
        plt.tight_layout(pad=.2)
        st.pyplot(fig, transparent=True)
        plt.close(fig)

        for cluster, count in segment_counts.items():
            name = cluster_names.get(cluster, f"Cluster {cluster}")
            pct = count / total_customers * 100
            html(f"""
            <div class="segment-row">
                <span><span class="segment-dot"></span>{name}</span>
                <strong>{pct:.2f}%</strong>
            </div>
            """)
        html('</div>')

    with overview_mid:
        html('<div class="dashboard-card"><div class="dashboard-card-title">📈 Annual Income vs Coverage Amount</div>')
        plot_df = df.sample(min(2200, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(5.2, 2.65), facecolor='none')
        ax.set_facecolor('none')
        scatter_colors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]
        for idx, cluster in enumerate(sorted(plot_df["Cluster"].unique())):
            sub = plot_df[plot_df["Cluster"] == cluster]
            ax.scatter(
                sub["Income Level"],
                sub["Coverage Amount"],
                s=9,
                alpha=.55,
                color=scatter_colors[idx % len(scatter_colors)],
                label=cluster_names.get(cluster, f"Cluster {cluster}")
            )
        ax.set_xlabel("Annual Income (₹)", fontsize=7, color="#94a3b8")
        ax.set_ylabel("Coverage Amount", fontsize=7, color="#94a3b8")
        ax.tick_params(labelsize=6, colors="#94a3b8")
        ax.grid(alpha=.12, color="white")
        leg = ax.legend(fontsize=6, loc="upper left", facecolor="#121826", edgecolor="#334155")
        for text in leg.get_texts():
            text.set_color("#f8fafc")
        plt.tight_layout(pad=.3)
        st.pyplot(fig, transparent=True)
        plt.close(fig)
        html('<div class="chart-note">Actual customer values from the segmentation dataset.</div></div>')

    with overview_right:
        html('<div class="dashboard-card"><div class="dashboard-card-title">⚙ Quick Filters</div>')
        filter_segment = st.selectbox(
            "Segment",
            ["All"] + sorted(df["Segment_Name"].dropna().unique().tolist()),
            key="quick_segment"
        )
        min_age = int(df["Age"].min())
        max_age = int(df["Age"].max())
        age_range = st.slider("Age", min_age, max_age, (min_age, max_age), key="quick_age")
        filter_gender = st.selectbox("Gender", ["All"] + sorted(df["Gender"].dropna().unique().tolist()), key="quick_gender")
        filter_education = st.selectbox("Education Level", ["All"] + sorted(df["Education Level"].dropna().unique().tolist()), key="quick_education")
        filter_marital = st.selectbox("Marital Status", ["All"] + sorted(df["Marital Status"].dropna().unique().tolist()), key="quick_marital")

        filtered = df[df["Age"].between(age_range[0], age_range[1])]
        if filter_segment != "All": filtered = filtered[filtered["Segment_Name"] == filter_segment]
        if filter_gender != "All": filtered = filtered[filtered["Gender"] == filter_gender]
        if filter_education != "All": filtered = filtered[filtered["Education Level"] == filter_education]
        if filter_marital != "All": filtered = filtered[filtered["Marital Status"] == filter_marital]

        st.button("⚱ Apply Filters", use_container_width=True, key="apply_filters")
        html(f'<div class="match-count">{len(filtered):,} matching customers</div></div>')

    # ---------------- INTELLIGENCE ----------------
    html('<div class="section dashboard-section">Customer Intelligence</div>')
    insight_col, prediction_col, recent_col = st.columns([1.0, 1.45, .92], gap="small")

    avg_income = df["Income Level"].mean()
    avg_coverage = df["Coverage Amount"].mean()
    avg_age = df["Age"].mean()
    avg_premium = df["Premium Amount"].mean()

    with insight_col:
        html('<div class="dashboard-card"><div class="dashboard-card-title">💡 Customer Insights</div>')
        a, b = st.columns(2, gap="small")
        with a:
            html(f'<div class="insight-mini"><div class="insight-icon green">₹</div><div class="mini-label">AVG. INCOME</div><div class="mini-value">₹{avg_income:,.0f}</div></div>')
        with b:
            html(f'<div class="insight-mini"><div class="insight-icon purple">▣</div><div class="mini-label">AVG. COVERAGE</div><div class="mini-value">₹{avg_coverage:,.0f}</div></div>')
        a, b = st.columns(2, gap="small")
        with a:
            html(f'<div class="insight-mini"><div class="insight-icon purple">👥</div><div class="mini-label">AVERAGE AGE</div><div class="mini-value">{avg_age:.1f} years</div></div>')
        with b:
            html(f'<div class="insight-mini"><div class="insight-icon teal">₹</div><div class="mini-label">AVG. PREMIUM</div><div class="mini-value">₹{avg_premium:,.0f}</div></div>')
        html('</div>')

    with prediction_col:
        html('<div class="dashboard-card"><div class="dashboard-card-title">🎯 Customer Prediction</div>')
        p1, p2, p3 = st.columns(3, gap="small")
        with p1:
            age = st.number_input("Age", 1, 100, 44, key="dash_age")
        with p2:
            gender = st.selectbox("Gender", ["Male", "Female"], key="dash_gender")
        with p3:
            marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"], key="dash_marital")
        p4, p5 = st.columns(2, gap="small")
        with p4:
            education = st.selectbox("Education Level", sorted(df["Education Level"].dropna().unique().tolist()), key="dash_education")
        with p5:
            occupation = st.text_input("Occupation", "Salesperson", key="dash_occupation")
        p6, p7 = st.columns(2, gap="small")
        with p6:
            income = st.number_input("Income Level", min_value=0.0, value=116579.06, step=1000.0, key="dash_income")
        with p7:
            coverage = st.number_input("Coverage Amount", min_value=0.0, value=478659.02, step=10000.0, key="dash_coverage")

        if st.button("✨ Predict Customer Segment", use_container_width=True, key="dash_predict"):
            customer = {
                "Age": age, "Gender": gender, "Marital Status": marital,
                "Education Level": education, "Geographic Information": "Lakshadweep",
                "Occupation": occupation, "Income Level": income,
                "Behavioral Data": "policy3", "Purchase History": "03-12-2018",
                "Interactions with Customer Service": "Chat", "Insurance Products Owned": "policy1",
                "Coverage Amount": coverage, "Premium Amount": 3000.0,
                "Policy Type": "Group", "Customer Preferences": "Mail",
                "Preferred Communication Channel": "In-Person Meeting",
                "Preferred Contact Time": "Morning", "Preferred Language": "German"
            }
            if not API_ONLINE:
                st.error("FastAPI is offline. Start the backend first.")
            else:
                try:
                    response = requests.post(f"{API_URL}/predict", json=customer, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.recent_predictions.insert(0, {
                            "Income": income,
                            "Age": age,
                            "Segment": result["segment_name"],
                            "Cluster": result["cluster"]
                        })
                        st.session_state.recent_predictions = st.session_state.recent_predictions[:5]
                        html(f'<div class="prediction-result"><div class="mini-label">PREDICTION RESULT</div><div class="prediction-name">{result["segment_name"]}</div><div class="prediction-sub">Customer belongs to Cluster {result["cluster"]}</div></div>')
                    else:
                        st.error(response.text)
                except Exception as e:
                    st.error(f"API error: {e}")
        html('</div>')

    with recent_col:
        html('<div class="dashboard-card"><div class="dashboard-card-title">🕘 Recent Predictions <span class="small-link">View All →</span></div>')
        if st.session_state.recent_predictions:
            st.dataframe(pd.DataFrame(st.session_state.recent_predictions), hide_index=True, use_container_width=True, height=205)
        else:
            html('<div class="empty-state">No predictions yet.<br><br>Use the prediction panel to create one.</div>')
        html('</div>')

    # ---------------- COMPARISON ----------------
    html('<div class="section dashboard-section">Segment Comparison</div>')
    c1, c2, c3 = st.columns(3, gap="small")

    chart_specs = [
        (c1, "Age Distribution", "Age", "#", 3.2),
        (c2, "Income Distribution", "Income Level", "₹", 3.2),
        (c3, "Coverage Distribution", "Coverage Amount", "₹", 3.2),
    ]

    for col, title, field, prefix, width in chart_specs:
        with col:
            html(f'<div class="dashboard-card"><div class="dashboard-card-title">{title}</div>')
            grouped = df.groupby("Segment_Name")[field].mean()
            fig, ax = plt.subplots(figsize=(3.25, 1.8), facecolor='none')
            ax.set_facecolor('none')
            bar_colors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b"]
            ax.bar(grouped.index, grouped.values, color=bar_colors[:len(grouped)], edgecolor="#334155", linewidth=1)
            ax.tick_params(axis="both", labelsize=6, colors="#94a3b8")
            ax.grid(axis="y", alpha=.12, color="white")
            plt.xticks(rotation=10, color="#cbd5e1")
            plt.yticks(color="#94a3b8")
            plt.tight_layout(pad=.2)
            st.pyplot(fig, transparent=True)
            plt.close(fig)
            html('</div>')


# ============================================================
# PROFILE PAGE
# ============================================================

elif page == "My Profile":

    st.markdown('<div class="page-title">My Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Manage your SegmentIQ workspace and view project information.</div>', unsafe_allow_html=True)

    st.markdown(
        """<div class="profile-shell">
            <div class="profile-avatar-large">P</div>
            <div class="profile-name-large">Pranav Rao</div>
            <div class="profile-role-large">Project User · Intelligent Customer Segmentation and Behavioural Analytics</div>
            <span class="profile-badge">● Active Workspace</span>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown("### Workspace Overview")
    p1, p2, p3, p4 = st.columns(4, gap="small")
    profile_cards = [
        (p1, "CUSTOMERS", f"{total_customers:,}", "Customers in the trained dataset"),
        (p2, "SEGMENTS", str(len(cluster_names)), "Discovered by K-Means clustering"),
        (p3, "MODEL", "K-Means", "Primary unsupervised algorithm"),
        (p4, "API STATUS", "Connected" if API_ONLINE else "Offline", "FastAPI prediction service"),
    ]
    for col, title, value, sub in profile_cards:
        with col:
            st.markdown(
                f'<div class="profile-info-card"><div class="profile-info-title">{title}</div><div class="profile-info-value">{value}</div><div class="profile-info-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("### Project Workspace")
    w1, w2 = st.columns(2, gap="small")
    with w1:
        st.markdown(
            """<div class="profile-action">
                <b>🎯 Your Role</b><br>
                <span style="font-size:12px;color:#64748b;line-height:1.6;">
                Project user working with customer segmentation, behavioural analytics,
                model predictions and customer exploration.
                </span>
            </div>""", unsafe_allow_html=True
        )
    with w2:
        st.markdown(
            f"""<div class="profile-action">
                <b>⚙ Current Model</b><br>
                <span style="font-size:12px;color:#64748b;line-height:1.6;">
                K-Means · {len(cluster_names)} clusters · Unsupervised learning ·
                PCA-based deployment pipeline.
                </span>
            </div>""", unsafe_allow_html=True
        )

    st.markdown("### Quick Actions")
    a1, a2, a3 = st.columns(3, gap="small")
    with a1:
        if st.button("🎯  Make a Prediction", key="profile_prediction", use_container_width=True):
            st.session_state.page_nav = "Customer Prediction"
            st.rerun()
    with a2:
        if st.button("📊  View Analytics", key="profile_analytics", use_container_width=True):
            st.session_state.page_nav = "Segment Analytics"
            st.rerun()
    with a3:
        if st.button("🔎  Explore Customers", key="profile_explorer", use_container_width=True):
            st.session_state.page_nav = "Customer Explorer"
            st.rerun()

    st.markdown("### System Information")
    s1, s2, s3 = st.columns(3, gap="small")
    with s1:
        st.markdown(f'<div class="profile-info-card"><div class="profile-info-title">Frontend</div><div class="profile-info-value">Streamlit</div><div class="profile-info-sub">Interactive dashboard and analytics UI</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="profile-info-card"><div class="profile-info-title">Backend</div><div class="profile-info-value">FastAPI</div><div class="profile-info-sub">Prediction API · {"Online" if API_ONLINE else "Offline"}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="profile-info-card"><div class="profile-info-title">Model Storage</div><div class="profile-info-value">Joblib</div><div class="profile-info-sub">Scikit-learn deployment artifact</div></div>', unsafe_allow_html=True)

    if st.button("←  Back to Dashboard", key="profile_back", use_container_width=False):
        st.session_state.page_nav = "Dashboard"
        st.rerun()


# ============================================================
# REMAINING PAGES
# ============================================================

elif page == "Customer Prediction":

    html("""
    <div class="section" style="font-size:22px;margin-top:5px;">
        Customer Prediction
    </div>
    <div class="muted" style="font-size:11px;margin-bottom:12px;">
        Enter customer information and predict the segment using the trained K-Means model.
    </div>
    """)

    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        full_age = st.number_input(
            "Age",
            1,
            100,
            44,
            key="full_age"
        )

    with c2:
        full_gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            key="full_gender"
        )

    with c3:
        full_marital = st.selectbox(
            "Marital Status",
            ["Married", "Single", "Divorced", "Widowed"],
            key="full_marital"
        )

    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        full_education = st.selectbox(
            "Education Level",
            [
                "Associate Degree",
                "Bachelor's Degree",
                "Master's Degree",
                "Doctorate",
                "High School"
            ],
            key="full_education"
        )

    with c2:
        full_geo = st.text_input(
            "Geographic Information",
            "Lakshadweep",
            key="full_geo"
        )

    with c3:
        full_occupation = st.text_input(
            "Occupation",
            "Salesperson",
            key="full_occupation"
        )

    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        full_income = st.number_input(
            "Income Level",
            min_value=0.0,
            value=89579.06,
            step=1000.0,
            key="full_income"
        )

    with c2:
        full_coverage = st.number_input(
            "Coverage Amount",
            min_value=0.0,
            value=500000.0,
            step=10000.0,
            key="full_coverage"
        )

    with c3:
        full_premium = st.number_input(
            "Premium Amount",
            min_value=0.0,
            value=3000.0,
            step=100.0,
            key="full_premium"
        )

    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        full_policy = st.selectbox(
            "Policy Type",
            ["Group", "Individual"],
            key="full_policy"
        )

    with c2:
        full_behavior = st.text_input(
            "Behavioral Data",
            "policy3",
            key="full_behavior"
        )

    with c3:
        full_insurance = st.text_input(
            "Insurance Products Owned",
            "policy1",
            key="full_insurance"
        )

    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        full_channel = st.selectbox(
            "Preferred Communication Channel",
            [
                "In-Person Meeting",
                "Email",
                "Phone",
                "SMS"
            ],
            key="full_channel"
        )

    with c2:
        full_contact = st.selectbox(
            "Preferred Contact Time",
            [
                "Morning",
                "Afternoon",
                "Evening",
                "Weekends"
            ],
            key="full_contact"
        )

    with c3:
        full_language = st.selectbox(
            "Preferred Language",
            [
                "German",
                "English",
                "French",
                "Spanish",
                "Hindi"
            ],
            key="full_language"
        )

    st.write("")

    if st.button(
        "✨ Predict Customer Segment",
        use_container_width=True,
        key="full_predict"
    ):

        customer = {
            "Age": full_age,
            "Gender": full_gender,
            "Marital Status": full_marital,
            "Education Level": full_education,
            "Geographic Information": full_geo,
            "Occupation": full_occupation,
            "Income Level": full_income,
            "Behavioral Data": full_behavior,
            "Purchase History": "03-12-2018",
            "Interactions with Customer Service": "Chat",
            "Insurance Products Owned": full_insurance,
            "Coverage Amount": full_coverage,
            "Premium Amount": full_premium,
            "Policy Type": full_policy,
            "Customer Preferences": "Mail",
            "Preferred Communication Channel": full_channel,
            "Preferred Contact Time": full_contact,
            "Preferred Language": full_language
        }

        if not API_ONLINE:
            st.error("FastAPI is offline.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=customer,
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()

                    html(f"""
                    <div class="result" style="margin-top:15px;">
                        <div class="result-label">PREDICTION RESULT</div>
                        <div class="result-name">{result["segment_name"]}</div>
                        <span class="pill">Cluster {result["cluster"]}</span>
                    </div>
                    """)
                else:
                    st.error(response.text)

            except Exception as e:
                st.error(f"API error: {e}")


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Segment Analytics":

    html("""
    <div class="section" style="font-size:22px;margin-top:5px;">
        Segment Analytics
    </div>
    <div class="muted" style="font-size:11px;margin-bottom:12px;">
        Detailed analysis of customer clusters discovered through unsupervised learning.
    </div>
    """)

    m1, m2, m3, m4 = st.columns(4, gap="small")

    m1.metric("Total Customers", f"{total_customers:,}")
    m2.metric("Segments", len(cluster_names))
    m3.metric("Higher-Income", f"{higher_pct:.2f}%")
    m4.metric("Lower-Income", f"{lower_pct:.2f}%")

    html('<div class="section">Customer Cluster Visualization</div>')

    sample = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    processed = preprocessor.transform(
        sample[feature_columns]
    )

    pca_visual = PCA(
        n_components=2,
        random_state=42
    )

    X_2d = pca_visual.fit_transform(
        processed
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.0), facecolor='none')
    ax.set_facecolor('none')
    pca_colors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b"]

    for idx, cluster in enumerate(sorted(sample["Cluster"].unique())):
        mask = (sample["Cluster"].values == cluster)
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=10,
            alpha=.55,
            color=pca_colors[idx % len(pca_colors)],
            label=cluster_names.get(cluster, f"Cluster {cluster}")
        )

    ax.set_xlabel("Principal Component 1", color="#94a3b8")
    ax.set_ylabel("Principal Component 2", color="#94a3b8")
    ax.set_title("Customer Clusters in PCA Space", color="#f8fafc", fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.grid(alpha=.12, color="white")
    leg = ax.legend(facecolor="#121826", edgecolor="#334155")
    for text in leg.get_texts():
        text.set_color("#f8fafc")

    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close(fig)

    html('<div class="section">Segment Financial Profile</div>')

    profile = (
        df.groupby("Segment_Name")[
            [
                "Age",
                "Income Level",
                "Coverage Amount",
                "Premium Amount"
            ]
        ]
        .mean()
        .round(2)
    )

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=False
    )

    html('<div class="section">Segment Charts</div>')

    chart1, chart2 = st.columns(2, gap="small")

    with chart1:
        html('<div class="card"><div class="card-title">Average Income by Segment</div>')
        income_data = df.groupby("Segment_Name")["Income Level"].mean()
        fig, ax = plt.subplots(figsize=(4.4, 2.4), facecolor='none')
        ax.set_facecolor('none')
        ax.bar(income_data.index, income_data.values, color=["#6366f1", "#06b6d4"], edgecolor="#334155")
        ax.set_ylabel("Average Income", fontsize=8, color="#94a3b8")
        ax.tick_params(axis="both", labelsize=7, colors="#94a3b8")
        ax.grid(axis="y", alpha=.12, color="white")
        plt.xticks(rotation=8, color="#cbd5e1")
        plt.yticks(color="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close(fig)
        html('</div>')

    with chart2:
        html('<div class="card"><div class="card-title">Average Coverage by Segment</div>')
        coverage_data = df.groupby("Segment_Name")["Coverage Amount"].mean()
        fig, ax = plt.subplots(figsize=(4.4, 2.4), facecolor='none')
        ax.set_facecolor('none')
        ax.bar(coverage_data.index, coverage_data.values, color=["#06b6d4", "#10b981"], edgecolor="#334155")
        ax.set_ylabel("Average Coverage", fontsize=8, color="#94a3b8")
        ax.tick_params(axis="both", labelsize=7, colors="#94a3b8")
        ax.grid(axis="y", alpha=.12, color="white")
        plt.xticks(rotation=8, color="#cbd5e1")
        plt.yticks(color="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close(fig)
        html('</div>')

    chart3, chart4 = st.columns(2, gap="small")

    with chart3:
        html('<div class="card"><div class="card-title">Average Age by Segment</div>')
        age_data = df.groupby("Segment_Name")["Age"].mean()
        fig, ax = plt.subplots(figsize=(4.4, 2.4), facecolor='none')
        ax.set_facecolor('none')
        ax.bar(age_data.index, age_data.values, color=["#10b981", "#f59e0b"], edgecolor="#334155")
        ax.set_ylabel("Average Age", fontsize=8, color="#94a3b8")
        ax.tick_params(axis="both", labelsize=7, colors="#94a3b8")
        ax.grid(axis="y", alpha=.12, color="white")
        plt.xticks(rotation=8, color="#cbd5e1")
        plt.yticks(color="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close(fig)
        html('</div>')

    with chart4:
        html('<div class="card"><div class="card-title">Average Premium by Segment</div>')
        premium_data = df.groupby("Segment_Name")["Premium Amount"].mean()
        fig, ax = plt.subplots(figsize=(4.4, 2.4), facecolor='none')
        ax.set_facecolor('none')
        ax.bar(premium_data.index, premium_data.values, color=["#a855f7", "#ec4899"], edgecolor="#334155")
        ax.set_ylabel("Average Premium", fontsize=8, color="#94a3b8")
        ax.tick_params(axis="both", labelsize=7, colors="#94a3b8")
        ax.grid(axis="y", alpha=.12, color="white")
        plt.xticks(rotation=8, color="#cbd5e1")
        plt.yticks(color="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close(fig)
        html('</div>')

    st.download_button(
        "⬇ Download Customer Segmentation Data",
        df.to_csv(index=False).encode("utf-8"),
        "customer_segments.csv",
        "text/csv",
        use_container_width=True
    )


# ============================================================
# CUSTOMER EXPLORER
# ============================================================

elif page == "Customer Explorer":

    st.markdown('<div class="page-title">Customer Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Search and explore customers with their assigned segments.</div>', unsafe_allow_html=True)

    search = st.text_input(
        "Search Customer ID",
        placeholder="Enter customer ID..."
    )

    segment_options = ["All"] + sorted(
        df["Segment_Name"].dropna().unique().tolist()
    )
    default_segment = st.session_state.pop("explorer_segment", "All")
    selected = st.selectbox(
        "Segment",
        segment_options,
        index=segment_options.index(default_segment) if default_segment in segment_options else 0,
        key="explorer_segment_select"
    )

    explorer = df.copy()

    if search:
        explorer = explorer[
            explorer["Customer ID"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    if selected != "All":
        explorer = explorer[
            explorer["Segment_Name"] == selected
        ]

    columns = [
        "Customer ID",
        "Age",
        "Gender",
        "Income Level",
        "Coverage Amount",
        "Premium Amount",
        "Segment_Name"
    ]

    st.dataframe(
        explorer[columns],
        use_container_width=True,
        hide_index=True,
        height=550
    )

    st.caption(
        f"Showing {len(explorer):,} customers"
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About Project":

    st.markdown('<div class="page-title">About SegmentIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Project overview, machine-learning workflow and deployment architecture.</div>', unsafe_allow_html=True)

    st.markdown(
        """<div class="about-hero">
            <div class="about-hero-title">Intelligent Customer Segmentation & Behavioural Analytics</div>
            <div class="about-hero-text">
                SegmentIQ uses unsupervised machine learning to discover meaningful customer segments
                from demographic, financial, behavioural and communication information, then exposes
                the trained model through a FastAPI service and an interactive Streamlit dashboard.
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown("### Machine Learning Pipeline")
    steps = ["Dataset", "Preprocessing", "Encoding + Scaling", "PCA", "K-Means", "Evaluation", "Profiling", "API + UI"]
    cols = st.columns(4, gap="small")
    for i, label in enumerate(steps):
        with cols[i % 4]:
            st.markdown(
                f'<div class="pipeline-step"><div class="pipeline-num">{i+1}</div><div class="pipeline-label">{label}</div></div>',
                unsafe_allow_html=True
            )
        if (i + 1) % 4 == 0 and i != len(steps) - 1:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.markdown("### Project Components")
    c1, c2, c3 = st.columns(3, gap="small")
    cards = [
        (c1, "🤖 Machine Learning", "K-Means is the primary clustering algorithm. Hierarchical clustering and DBSCAN were used as comparison approaches."),
        (c2, "📈 Evaluation", "The project evaluates clustering with Silhouette Score, Davies-Bouldin Index and Calinski-Harabasz Index."),
        (c3, "🌐 Application", "Streamlit provides the interactive dashboard while FastAPI serves real-time customer segment predictions."),
    ]
    for col, title, text in cards:
        with col:
            st.markdown(f'<div class="about-card"><div class="about-card-title">{title}</div><div class="about-card-text">{text}</div></div>', unsafe_allow_html=True)

    st.markdown("### Current Model Summary")
    m1, m2 = st.columns([1, 2], gap="small")
    with m1:
        st.metric("Customers", f"{total_customers:,}")
        st.metric("Clusters", len(cluster_names))
    with m2:
        st.markdown(
            f"""<div class="about-card" style="min-height:190px;">
                <div class="about-card-title">Discovered Segments</div>
                <div class="about-card-text">
                    <b style="color:#a5b4fc;">Higher-Income Customers</b> — {higher_count:,} customers ({higher_pct:.2f}%)<br><br>
                    <b style="color:#a5b4fc;">Lower-Income Customers</b> — {lower_count:,} customers ({lower_pct:.2f}%)<br><br>
                    Segment names are based on the observed income-level difference in the final cluster profiles.
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("### Technology Stack")
    tech = ["Python", "Pandas", "NumPy", "Scikit-learn", "Joblib", "PCA", "K-Means", "FastAPI", "Streamlit", "Matplotlib", "Google Colab", "Kaggle"]
    st.markdown(" ".join([f'<span class="tech-pill">{x}</span>' for x in tech]), unsafe_allow_html=True)

    st.markdown("### Deployment Architecture")
    st.code(
        "Kaggle Dataset / Model Training\n"
        "        ↓\n"
        "Scikit-learn Model (.pkl)\n"
        "        ↓\n"
        "FastAPI Prediction Service\n"
        "        ↓\n"
        "Streamlit Interactive Dashboard\n"
        "        ↓\n"
        "Customer Prediction + Analytics + Explorer",
        language="text"
    )

    with st.expander("📌 What makes the project unsupervised?", expanded=False):
        st.write(
            "The existing 'Segmentation Group' column is excluded from model training. "
            "K-Means discovers groups from the available customer features without using a target label."
        )

    with st.expander("📊 Why these evaluation metrics?", expanded=False):
        st.write(
            "Clustering is evaluated with internal clustering metrics rather than classification metrics. "
            "Silhouette measures separation and cohesion, Davies-Bouldin evaluates cluster similarity, "
            "and Calinski-Harabasz compares between-cluster and within-cluster dispersion."
        )


# ============================================================
# FOOTER
# ============================================================

html("""
<div class="footer">
    SegmentIQ • Intelligent Customer Segmentation and Behavioural Analytics
</div>
""")
