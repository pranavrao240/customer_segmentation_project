
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

    /* Sidebar Navigation Radios */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 6px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius: 12px;
        padding: 9px 12px;
        background: transparent;
        transition: all .15s ease;
        color: #94a3b8;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #ffffff;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
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

    /* Section titles */
    .section {
        font-size: 18px !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        margin: 16px 0 10px 1px;
        letter-spacing: -0.3px;
    }

    /* KPI */
    .kpi {
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

    .kpi:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4) !important;
    }

    .kpi.blue {
        border-left: 3px solid #3b82f6 !important;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(18, 24, 38, 0.75) 100%) !important;
    }

    .kpi.purple {
        border-left: 3px solid #a855f7 !important;
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.12) 0%, rgba(18, 24, 38, 0.75) 100%) !important;
    }

    .kpi.green {
        border-left: 3px solid #10b981 !important;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(18, 24, 38, 0.75) 100%) !important;
    }

    .kpi.pink {
        border-left: 3px solid #f43f5e !important;
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.12) 0%, rgba(18, 24, 38, 0.75) 100%) !important;
    }

    .kpi-label {
        font-size: 11px !important;
        color: #94a3b8 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    .kpi-value {
        color: #f8fafc !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin-top: 5px;
    }

    .kpi-sub {
        color: #64748b !important;
        font-size: 11px !important;
        margin-top: 4px;
    }

    /* Cards */
    .card {
        background: rgba(18, 24, 38, 0.72) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3) !important;
        height: 100%;
        transition: border-color .2s ease;
    }

    .card:hover {
        border-color: rgba(99, 102, 241, 0.3) !important;
    }

    .card-title {
        color: #f1f5f9 !important;
        font-size: 14px !important;
        font-weight: 800 !important;
        letter-spacing: -0.2px;
        margin-bottom: 12px !important;
    }

    .muted {
        color: #94a3b8 !important;
        font-size: 11px !important;
    }

    /* Mini insight cards */
    .mini {
        min-height: 80px;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        margin-bottom: 8px;
        background: rgba(25, 33, 50, 0.6) !important;
        backdrop-filter: blur(10px);
    }

    .mini-label {
        color: #94a3b8 !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .mini-value {
        color: #f8fafc !important;
        font-size: 13px !important;
        font-weight: 700;
        margin-top: 3px;
    }

    /* Prediction */
    .result {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(6, 78, 59, 0.25)) !important;
        border: 1px solid rgba(16, 185, 129, 0.35) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        margin-top: 10px;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.14) !important;
    }

    .result-label {
        color: #a7f3d0 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .result-name {
        color: #34d399 !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        margin-top: 4px;
        letter-spacing: -0.3px;
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

    /* Inputs */
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label,
    .stSlider label {
        color: #cbd5e1 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    .stTextInput input,
    .stNumberInput input {
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

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
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

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 11px;
        padding-top: 24px;
    }

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

API_URL = "http://127.0.0.1:8000"


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

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "👤 Customer Prediction",
            "📊 Segment Analytics",
            "🔎 Customer Explorer",
            "ℹ️ About Project",
        ],
        label_visibility="collapsed",
    )

    html('<div class="side-title">MODEL STATUS</div>')

    if API_ONLINE:
        html('<div style="color:#5ee2a8;font-size:10px;font-weight:700;">● FastAPI Connected</div>')
    else:
        html('<div style="color:#ff7f8d;font-size:10px;font-weight:700;">● FastAPI Offline</div>')

    html("""
    <div class="side-title">MODEL</div>
    <div style="font-size:9px;color:#93a4c6;">Algorithm</div>
    <div style="font-size:11px;font-weight:700;margin-bottom:8px;">K-Means</div>
    <div style="font-size:9px;color:#93a4c6;">Learning</div>
    <div style="font-size:11px;font-weight:700;margin-bottom:8px;">Unsupervised</div>
    <div style="font-size:9px;color:#93a4c6;">Clusters</div>
    <div style="font-size:11px;font-weight:700;">2</div>

    <hr style="border:0;border-top:1px solid #243556;margin:22px 0;">

    <div style="font-size:9px;color:#91a1c1;line-height:1.6;">
        Intelligent Customer Segmentation<br>
        and Behavioural Analytics
    </div>
    """)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    # Functional dashboard search + profile area
    search_col, notify_col, profile_col = st.columns([8, 1, 2], gap="small")

    with search_col:
        dashboard_search = st.text_input(
            "",
            placeholder="🔍  Search customer ID, segment, gender, occupation...",
            label_visibility="collapsed",
            key="dashboard_search"
        )

    with notify_col:
        st.markdown("<div style='padding-top:7px;text-align:center;font-size:20px;'>🔔</div>", unsafe_allow_html=True)

    with profile_col:
        profile_open = st.button("👤  Pranav Rao", use_container_width=True, key="profile_button")

    if profile_open:
        st.info("Profile: Pranav Rao\n\nRole: Project User\n\nApplication: Intelligent Customer Segmentation and Behavioural Analytics")

    if dashboard_search:
        q = dashboard_search.strip().lower()
        search_mask = (
            df["Customer ID"].astype(str).str.lower().str.contains(q, na=False) |
            df["Segment_Name"].astype(str).str.lower().str.contains(q, na=False) |
            df["Gender"].astype(str).str.lower().str.contains(q, na=False) |
            df["Occupation"].astype(str).str.lower().str.contains(q, na=False)
        )
        search_results = df.loc[search_mask].head(10)

        if not search_results.empty:
            st.markdown("### 🔎 Search Results")
            st.dataframe(
                search_results[[
                    "Customer ID", "Age", "Gender", "Occupation",
                    "Income Level", "Coverage Amount", "Premium Amount",
                    "Segment_Name"
                ]],
                use_container_width=True,
                hide_index=True,
                height=min(320, 80 + len(search_results) * 35)
            )

            selected_id = st.selectbox(
                "View customer profile",
                search_results["Customer ID"].astype(str).tolist(),
                key="dashboard_profile_customer"
            )
            customer_row = search_results[
                search_results["Customer ID"].astype(str) == str(selected_id)
            ].iloc[0]

            profile_a, profile_b, profile_c, profile_d = st.columns(4)
            profile_a.metric("Age", int(customer_row["Age"]))
            profile_b.metric("Income", f"{customer_row['Income Level']:,.0f}")
            profile_c.metric("Coverage", f"{customer_row['Coverage Amount']:,.0f}")
            profile_d.metric("Segment", customer_row["Segment_Name"])
        else:
            st.warning("No matching customer or segment found.")

    html("""
    <div class="hero">
        <div class="hero-greeting">👋 Hello, Pranav!</div>
        <div class="hero-title">Welcome to Customer Intelligence Dashboard</div>
        <div class="hero-desc">
            Explore customer segments, understand behavioural patterns
            and predict customer groups using unsupervised machine learning.
        </div>
    </div>
    """)

    st.write("")

    k1, k2, k3, k4 = st.columns(
        4,
        gap="small"
    )

    with k1:
        html(f"""
        <div class="kpi blue">
            <div class="kpi-label">👥 &nbsp; Total Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-sub">Customers in dataset</div>
        </div>
        """)

    with k2:
        html(f"""
        <div class="kpi purple">
            <div class="kpi-label">🧩 &nbsp; Customer Segments</div>
            <div class="kpi-value">{len(cluster_names)}</div>
            <div class="kpi-sub">Discovered by K-Means</div>
        </div>
        """)

    with k3:
        html(f"""
        <div class="kpi green">
            <div class="kpi-label">↑ &nbsp; Higher-Income Customers</div>
            <div class="kpi-value">{higher_pct:.2f}%</div>
            <div class="kpi-sub">{higher_count:,} customers</div>
        </div>
        """)

    with k4:
        html(f"""
        <div class="kpi pink">
            <div class="kpi-label">↓ &nbsp; Lower-Income Customers</div>
            <div class="kpi-value">{lower_pct:.2f}%</div>
            <div class="kpi-sub">{lower_count:,} customers</div>
        </div>
        """)

    html('<div class="section">Customer Overview</div>')

    left, middle, right = st.columns(
        [1.02, 1.55, .85],
        gap="small"
    )

    # Segment card
    with left:

        html("""
        <div class="card">
            <div class="card-title">Customer Segments</div>
        """)

        fig, ax = plt.subplots(
            figsize=(3.4, 2.2), facecolor='none'
        )
        ax.set_facecolor('none')
        donut_colors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]

        values = segment_counts.values
        labels = [
            cluster_names.get(i, f"Cluster {i}")
            for i in segment_counts.index
        ]

        ax.pie(
            values,
            labels=None,
            autopct="%1.1f%%",
            startangle=90,
            colors=donut_colors[:len(values)],
            wedgeprops={
                "width": .42,
                "edgecolor": "#0d1322",
                "linewidth": 2
            },
            textprops={"fontsize": 8, "color": "#f1f5f9", "weight": "bold"}
        )

        ax.set_title(
            "Segment Distribution",
            fontsize=10,
            fontweight="bold",
            color="#f8fafc"
        )

        plt.tight_layout()
        st.pyplot(fig, width=700, transparent=True)
        plt.close(fig)

        for cluster, count in segment_counts.items():
            pct = count / total_customers * 100
            name = cluster_names.get(
                cluster,
                f"Cluster {cluster}"
            )

            html(f"""
            <div style="
                display:flex;
                justify-content:space-between;
                padding:3px 0;
                font-size:9px;
                color:#53627a;
            ">
                <span>{name}</span>
                <b>{pct:.2f}%</b>
            </div>
            """)

        html("</div>")

    # Scatter
    with middle:

        html("""
        <div class="card">
            <div class="card-title">Income vs Coverage Amount</div>
        """)

        plot_df = df.sample(
            min(3500, len(df)),
            random_state=42
        )

        fig, ax = plt.subplots(
            figsize=(5.2, 2.3), facecolor='none'
        )
        ax.set_facecolor('none')
        scatter_colors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]

        for idx, cluster in enumerate(sorted(
            plot_df["Cluster"].unique()
        )):
            subset = plot_df[
                plot_df["Cluster"] == cluster
            ]

            ax.scatter(
                subset["Income Level"],
                subset["Coverage Amount"],
                s=8,
                alpha=.55,
                color=scatter_colors[idx % len(scatter_colors)],
                label=cluster_names.get(
                    cluster,
                    f"Cluster {cluster}"
                )
            )

        ax.set_xlabel(
            "Income Level",
            fontsize=8,
            color="#94a3b8"
        )
        ax.set_ylabel(
            "Coverage Amount",
            fontsize=8,
            color="#94a3b8"
        )
        ax.tick_params(labelsize=7, colors="#94a3b8")
        ax.grid(alpha=.12, color="white")
        leg = ax.legend(fontsize=7, facecolor="#121826", edgecolor="#334155")
        for text in leg.get_texts():
            text.set_color("#f8fafc")

        plt.tight_layout()
        st.pyplot(fig, width=700, transparent=True)
        plt.close(fig)

        html("""
        <div class="muted">
            Actual customer values from the project dataset.
        </div>
        </div>
        """)

    # Filters
    with right:

        html("""
        <div class="card">
            <div class="card-title">⚙ Quick Filters</div>
        """)

        selected_segment = st.selectbox(
            "Segment",
            ["All"] + sorted(
                df["Segment_Name"]
                .dropna()
                .unique()
                .tolist()
            ),
            key="dashboard_segment"
        )

        min_income = float(df["Income Level"].min())
        max_income = float(df["Income Level"].max())

        income_range = st.slider(
            "Income Range",
            min_value=min_income,
            max_value=max_income,
            value=(min_income, max_income),
            key="dashboard_income_range"
        )

        filtered = df[
            df["Income Level"].between(
                income_range[0],
                income_range[1]
            )
        ]

        if selected_segment != "All":
            filtered = filtered[
                filtered["Segment_Name"] == selected_segment
            ]

        html(f"""
        <div class="mini" style="margin-top:9px;">
            <div class="mini-label">MATCHING CUSTOMERS</div>
            <div class="mini-value">{len(filtered):,}</div>
        </div>
        </div>
        """)

    html('<div class="section">Customer Intelligence</div>')

    insight_col, prediction_col, recent_col = st.columns(
        [1.0, 1.45, 1.0],
        gap="small"
    )

    # Insights
    with insight_col:

        html("""
        <div class="card">
            <div class="card-title">💡 Customer Insights</div>
        """)

        avg_income = df["Income Level"].mean()
        avg_coverage = df["Coverage Amount"].mean()
        avg_age = df["Age"].mean()
        avg_premium = df["Premium Amount"].mean()

        a, b = st.columns(2, gap="small")

        with a:
            html(f"""
            <div class="mini">
                <div class="mini-label">AVG. INCOME</div>
                <div class="mini-value">₹{avg_income:,.0f}</div>
            </div>
            """)

        with b:
            html(f"""
            <div class="mini">
                <div class="mini-label">AVG. COVERAGE</div>
                <div class="mini-value">₹{avg_coverage:,.0f}</div>
            </div>
            """)

        st.write("")

        a, b = st.columns(2, gap="small")

        with a:
            html(f"""
            <div class="mini">
                <div class="mini-label">AVG. AGE</div>
                <div class="mini-value">{avg_age:.1f} years</div>
            </div>
            """)

        with b:
            html(f"""
            <div class="mini">
                <div class="mini-label">AVG. PREMIUM</div>
                <div class="mini-value">₹{avg_premium:,.0f}</div>
            </div>
            """)

        html("</div>")

    # Prediction
    with prediction_col:

        html("""
        <div class="card">
            <div class="card-title">🎯 Customer Prediction</div>
        """)

        p1, p2, p3 = st.columns(3, gap="small")

        with p1:
            age = st.number_input(
                "Age",
                1,
                100,
                44,
                key="dash_age"
            )

        with p2:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                key="dash_gender"
            )

        with p3:
            marital = st.selectbox(
                "Marital Status",
                ["Married", "Single", "Divorced", "Widowed"],
                key="dash_marital"
            )

        p4, p5 = st.columns(2, gap="small")

        with p4:
            education = st.selectbox(
                "Education Level",
                [
                    "Associate Degree",
                    "Bachelor's Degree",
                    "Master's Degree",
                    "Doctorate",
                    "High School"
                ],
                key="dash_education"
            )

        with p5:
            occupation = st.text_input(
                "Occupation",
                "Salesperson",
                key="dash_occupation"
            )

        p6, p7 = st.columns(2, gap="small")

        with p6:
            income = st.number_input(
                "Income Level",
                min_value=0.0,
                value=89579.06,
                step=1000.0,
                key="dash_income"
            )

        with p7:
            coverage = st.number_input(
                "Coverage Amount",
                min_value=0.0,
                value=500000.0,
                step=10000.0,
                key="dash_coverage"
            )

        if st.button(
            "✨ Predict Customer Segment",
            use_container_width=True,
            key="dash_predict"
        ):

            customer = {
                "Age": age,
                "Gender": gender,
                "Marital Status": marital,
                "Education Level": education,
                "Geographic Information": "Lakshadweep",
                "Occupation": occupation,
                "Income Level": income,
                "Behavioral Data": "policy3",
                "Purchase History": "03-12-2018",
                "Interactions with Customer Service": "Chat",
                "Insurance Products Owned": "policy1",
                "Coverage Amount": coverage,
                "Premium Amount": 3000.0,
                "Policy Type": "Group",
                "Customer Preferences": "Mail",
                "Preferred Communication Channel": "In-Person Meeting",
                "Preferred Contact Time": "Morning",
                "Preferred Language": "German"
            }

            if not API_ONLINE:
                st.error(
                    "FastAPI is offline. Start the backend first."
                )
            else:
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=customer,
                        timeout=10
                    )

                    if response.status_code == 200:
                        result = response.json()

                        st.session_state.recent_predictions.insert(
                            0,
                            {
                                "Age": age,
                                "Income": income,
                                "Segment": result["segment_name"],
                                "Cluster": result["cluster"]
                            }
                        )

                        st.session_state.recent_predictions = (
                            st.session_state.recent_predictions[:5]
                        )

                        html(f"""
                        <div class="result">
                            <div class="result-label">PREDICTION RESULT</div>
                            <div class="result-name">{result["segment_name"]}</div>
                            <span class="pill">Cluster {result["cluster"]}</span>
                        </div>
                        """)

                    else:
                        st.error(response.text)

                except Exception as e:
                    st.error(f"API error: {e}")

        html("</div>")

    # Recent
    with recent_col:

        html("""
        <div class="card">
            <div class="card-title">🕘 Recent Predictions</div>
        """)

        if st.session_state.recent_predictions:

            recent = pd.DataFrame(
                st.session_state.recent_predictions
            )

            st.dataframe(
                recent,
                hide_index=True,
                use_container_width=True,
                height=205
            )

        else:

            html("""
            <div style="
                text-align:center;
                color:#8997ad;
                font-size:10px;
                padding:45px 8px;
            ">
                No predictions yet.<br><br>
                Use the prediction panel to create one.
            </div>
            """)

        html("</div>")

    html('<div class="section">Segment Comparison</div>')

    c1, c2 = st.columns(2, gap="small")

    with c1:

        html("""
        <div class="card">
            <div class="card-title">Average Income by Segment</div>
        """)

        income_data = (
            df.groupby("Segment_Name")["Income Level"]
            .mean()
        )

        fig, ax = plt.subplots(figsize=(4.6, 2.0), facecolor='none')
        ax.set_facecolor('none')

        ax.bar(
            income_data.index,
            income_data.values,
            color=["#6366f1", "#06b6d4"],
            edgecolor="#334155"
        )

        ax.set_ylabel(
            "Average Income",
            fontsize=8,
            color="#94a3b8"
        )
        ax.tick_params(labelsize=7, colors="#94a3b8")
        ax.grid(axis="y", alpha=.12, color="white")
        plt.xticks(rotation=8, color="#cbd5e1")
        plt.yticks(color="#94a3b8")
        plt.tight_layout()

        st.pyplot(fig, width=700, transparent=True)
        plt.close(fig)

        html("</div>")

    with c2:

        html("""
        <div class="card">
            <div class="card-title">Average Coverage by Segment</div>
        """)

        coverage_data = (
            df.groupby("Segment_Name")["Coverage Amount"]
            .mean()
        )

        fig, ax = plt.subplots(figsize=(4.6, 2.0), facecolor='none')
        ax.set_facecolor('none')

        ax.bar(
            coverage_data.index,
            coverage_data.values,
            color=["#06b6d4", "#10b981"],
            edgecolor="#334155"
        )

        ax.set_ylabel(
            "Average Coverage",
            fontsize=8,
            color="#94a3b8"
        )
        ax.tick_params(labelsize=7, colors="#94a3b8")
        ax.grid(axis="y", alpha=.12, color="white")
        plt.xticks(rotation=8, color="#cbd5e1")
        plt.yticks(color="#94a3b8")
        plt.tight_layout()

        st.pyplot(fig, width=700, transparent=True)
        plt.close(fig)

        html("</div>")


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

elif page == "👤 Customer Prediction":

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

elif page == "📊 Segment Analytics":

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

    fig, ax = plt.subplots(
        figsize=(6.2, 3.0), facecolor='none'
    )
    ax.set_facecolor('none')
    pca_colors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b"]

    for idx, cluster in enumerate(sorted(
        sample["Cluster"].unique()
    )):
        mask = (
            sample["Cluster"].values == cluster
        )

        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=10,
            alpha=.55,
            color=pca_colors[idx % len(pca_colors)],
            label=cluster_names.get(
                cluster,
                f"Cluster {cluster}"
            )
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
    st.pyplot(fig, width=700, transparent=True)
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
        st.pyplot(fig, width=520, transparent=True)
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
        st.pyplot(fig, width=520, transparent=True)
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
        st.pyplot(fig, width=520, transparent=True)
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
        st.pyplot(fig, width=520, transparent=True)
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

elif page == "🔎 Customer Explorer":

    html("""
    <div class="section" style="font-size:22px;margin-top:5px;">
        Customer Explorer
    </div>
    <div class="muted" style="font-size:11px;margin-bottom:12px;">
        Search and explore customers with their assigned segments.
    </div>
    """)

    search = st.text_input(
        "Search Customer ID",
        placeholder="Enter customer ID..."
    )

    selected = st.selectbox(
        "Segment",
        ["All"] + sorted(
            df["Segment_Name"]
            .dropna()
            .unique()
            .tolist()
        )
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

elif page == "ℹ️ About Project":

    st.markdown("""<div class="section" style="font-size:22px;margin-top:5px;">About SegmentIQ</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="card">
<div class="card-title">Intelligent Customer Segmentation and Behavioural Analytics</div>
<p style="font-size:12px;color:#94a3b8;line-height:1.7;margin:8px 0 18px 0;">This project applies unsupervised machine learning to discover customer segments from demographic, financial, behavioural and communication information.</p>
<div style="font-weight:700;color:#f8fafc;font-size:12px;margin-bottom:7px;">Machine Learning Pipeline</div>
<div style="padding:12px;background:rgba(25, 33, 50, 0.6);border:1px solid rgba(255,255,255,0.08);border-radius:9px;font-size:11px;color:#cbd5e1;line-height:1.7;">Dataset → Preprocessing → Encoding + Scaling → PCA → K-Means → Cluster Evaluation → Segment Profiling → FastAPI → Streamlit Dashboard</div>
<div style="font-weight:700;color:#f8fafc;font-size:12px;margin:18px 0 7px 0;">Algorithms</div>
<div style="font-size:11px;color:#94a3b8;line-height:1.8;">• K-Means Clustering — primary algorithm<br>• Hierarchical / Agglomerative Clustering — comparison<br>• DBSCAN — comparison</div>
<div style="font-weight:700;color:#f8fafc;font-size:12px;margin:18px 0 7px 0;">Evaluation Metrics</div>
<div style="font-size:11px;color:#94a3b8;line-height:1.8;">• Silhouette Score<br>• Davies-Bouldin Index<br>• Calinski-Harabasz Index</div>
<div style="font-weight:700;color:#f8fafc;font-size:12px;margin:18px 0 7px 0;">Technology Stack</div>
<div style="font-size:11px;color:#94a3b8;line-height:1.8;">Python • Pandas • NumPy • Scikit-learn • Joblib • FastAPI • Streamlit • Matplotlib • Google Colab • Kaggle</div>
</div>""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

html("""
<div class="footer">
    SegmentIQ • Intelligent Customer Segmentation and Behavioural Analytics
</div>
""")
