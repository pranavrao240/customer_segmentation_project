import streamlit as st
import pandas as pd
import requests
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #888888;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# FILE PATHS
# ==================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "customer_segments.csv"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "backend",
    "model",
    "customer_segmentation_model.pkl"
)


# ==================================================
# LOAD DATASET
# ==================================================

try:

    df = pd.read_csv(DATA_PATH)

    data_loaded = True

except Exception:

    df = None

    data_loaded = False


# ==================================================
# LOAD MODEL
# ==================================================

try:

    model_data = joblib.load(MODEL_PATH)

    model_loaded = True

except Exception:

    model_data = None

    model_loaded = False


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("📊 Customer Segmentation")

st.sidebar.write(
    "Intelligent Customer Segmentation "
    "and Behavioural Analytics"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Customer Prediction",
        "📈 Segment Analytics",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Machine Learning Model\n\n"
    "Algorithm: K-Means Clustering\n"
    "Learning Type: Unsupervised Learning"
)


# ==================================================
# PAGE 1 — CUSTOMER PREDICTION
# ==================================================

if page == "🏠 Customer Prediction":

    st.markdown(
        '<div class="main-title">'
        '🏠 Customer Segment Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter customer information to identify the '
        'corresponding customer segment using K-Means clustering.'
        '</div>',
        unsafe_allow_html=True
    )


    # ==================================================
    # PERSONAL INFORMATION
    # ==================================================

    st.markdown(
        '<div class="section-title">'
        '👤 Personal Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=44
        )


    with col2:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )


    with col3:

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Divorced",
                "Widowed"
            ]
        )


    with col4:

        education_level = st.selectbox(
            "Education Level",
            [
                "High School",
                "Bachelor's Degree",
                "Master's Degree",
                "Doctorate",
                "Associate Degree"
            ]
        )


    st.divider()


    # ==================================================
    # LOCATION & OCCUPATION
    # ==================================================

    st.markdown(
        '<div class="section-title">'
        '📍 Location & Occupation'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")


    col1, col2, col3 = st.columns(3)


    with col1:

        geographic_information = st.text_input(
            "Geographic Information",
            value="Lakshadweep"
        )


    with col2:

        occupation = st.text_input(
            "Occupation",
            value="Salesperson"
        )


    with col3:

        preferred_language = st.text_input(
            "Preferred Language",
            value="German"
        )


    st.divider()


    # ==================================================
    # FINANCIAL INFORMATION
    # ==================================================

    st.markdown(
        '<div class="section-title">'
        '💰 Financial Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")


    col1, col2, col3 = st.columns(3)


    with col1:

        income_level = st.number_input(
            "Income Level",
            min_value=0.0,
            value=116579.06,
            step=1000.0
        )


    with col2:

        coverage_amount = st.number_input(
            "Coverage Amount",
            min_value=0.0,
            value=478659.02,
            step=1000.0
        )


    with col3:

        premium_amount = st.number_input(
            "Premium Amount",
            min_value=0.0,
            value=3019.73,
            step=100.0
        )


    st.divider()


    # ==================================================
    # POLICY INFORMATION
    # ==================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Policy Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")


    col1, col2, col3 = st.columns(3)


    with col1:

        policy_type = st.text_input(
            "Policy Type",
            value="Group"
        )


    with col2:

        insurance_products = st.text_input(
            "Insurance Products Owned",
            value="policy1"
        )


    with col3:

        customer_preferences = st.text_input(
            "Customer Preferences",
            value="Mail"
        )


    st.divider()


    # ==================================================
    # BEHAVIOURAL INFORMATION
    # ==================================================

    st.markdown(
        '<div class="section-title">'
        '🧠 Behavioural Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")


    col1, col2, col3 = st.columns(3)


    with col1:

        behavioral_data = st.text_input(
            "Behavioral Data",
            value="policy3"
        )


    with col2:

        purchase_history = st.text_input(
            "Purchase History",
            value="03-12-2018"
        )


    with col3:

        customer_service = st.selectbox(
            "Interactions with Customer Service",
            [
                "Chat",
                "Email",
                "Phone",
                "In-Person"
            ]
        )


    st.divider()


    # ==================================================
    # COMMUNICATION INFORMATION
    # ==================================================

    st.markdown(
        '<div class="section-title">'
        '📞 Communication Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")


    col1, col2 = st.columns(2)


    with col1:

        communication_channel = st.text_input(
            "Preferred Communication Channel",
            value="In-Person Meeting"
        )


    with col2:

        contact_time = st.text_input(
            "Preferred Contact Time",
            value="Weekends"
        )


    st.divider()


    # ==================================================
    # PREDICT BUTTON
    # ==================================================

    st.write("")


    predict_button = st.button(
        "🔍 Predict Customer Segment",
        use_container_width=True
    )


    if predict_button:

        customer_data = {

            "Age":
                age,

            "Gender":
                gender,

            "Marital Status":
                marital_status,

            "Education Level":
                education_level,

            "Geographic Information":
                geographic_information,

            "Occupation":
                occupation,

            "Income Level":
                income_level,

            "Behavioral Data":
                behavioral_data,

            "Purchase History":
                purchase_history,

            "Interactions with Customer Service":
                customer_service,

            "Insurance Products Owned":
                insurance_products,

            "Coverage Amount":
                coverage_amount,

            "Premium Amount":
                premium_amount,

            "Policy Type":
                policy_type,

            "Customer Preferences":
                customer_preferences,

            "Preferred Communication Channel":
                communication_channel,

            "Preferred Contact Time":
                contact_time,

            "Preferred Language":
                preferred_language
        }


        # ==========================================
        # SEND REQUEST TO FASTAPI
        # ==========================================

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=customer_data,
                timeout=10
            )


            if response.status_code == 200:

                result = response.json()

                cluster = result["cluster"]

                segment_name = result["segment_name"]


                st.divider()


                # ==================================
                # PREDICTION RESULT
                # ==================================

                st.subheader(
                    "🎯 Prediction Result"
                )


                st.success(
                    "Customer segmentation completed successfully!"
                )


                result_col1, result_col2 = \
                    st.columns(2)


                with result_col1:

                    st.metric(
                        "Predicted Cluster",
                        cluster
                    )


                with result_col2:

                    st.metric(
                        "Customer Segment",
                        segment_name
                    )


                st.divider()


                # ==================================
                # CUSTOMER PROFILE
                # ==================================

                st.subheader(
                    "👤 Customer Profile"
                )


                profile_col1, profile_col2, profile_col3 = \
                    st.columns(3)


                with profile_col1:

                    st.write(
                        f"**Age:** {age}"
                    )

                    st.write(
                        f"**Gender:** {gender}"
                    )

                    st.write(
                        f"**Marital Status:** "
                        f"{marital_status}"
                    )

                    st.write(
                        f"**Education:** "
                        f"{education_level}"
                    )

                    st.write(
                        f"**Occupation:** "
                        f"{occupation}"
                    )


                with profile_col2:

                    st.write(
                        f"**Income:** "
                        f"₹{income_level:,.2f}"
                    )

                    st.write(
                        f"**Coverage:** "
                        f"₹{coverage_amount:,.2f}"
                    )

                    st.write(
                        f"**Premium:** "
                        f"₹{premium_amount:,.2f}"
                    )

                    st.write(
                        f"**Policy Type:** "
                        f"{policy_type}"
                    )


                with profile_col3:

                    st.write(
                        f"**Behavior:** "
                        f"{behavioral_data}"
                    )

                    st.write(
                        f"**Communication:** "
                        f"{communication_channel}"
                    )

                    st.write(
                        f"**Contact Time:** "
                        f"{contact_time}"
                    )

                    st.write(
                        f"**Language:** "
                        f"{preferred_language}"
                    )


            else:

                st.error(
                    "Prediction failed."
                )

                st.code(
                    response.text
                )


        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI backend."
            )

            st.info(
                "Please make sure Terminal 1 "
                "is running FastAPI."
            )


        except Exception as e:

            st.error(
                "An unexpected error occurred."
            )

            st.code(
                str(e)
            )


# ==================================================
# PAGE 2 — SEGMENT ANALYTICS
# ==================================================

elif page == "📈 Segment Analytics":

    st.title(
        "📈 Customer Segment Analytics"
    )

    st.write(
        "Analysis of customer segments generated "
        "using K-Means clustering."
    )


    if not data_loaded:

        st.error(
            "customer_segments.csv could not be loaded."
        )

        st.info(
            "Make sure the CSV is inside the "
            "frontend folder."
        )


    else:

        # ==========================================
        # DATASET INFORMATION
        # ==========================================

        total_customers = len(df)

        number_of_clusters = \
            df["Cluster"].nunique()


        segment_percentages = (
            df["Segment_Name"]
            .value_counts(
                normalize=True
            )
            * 100
        )


        higher_income_percentage = \
            segment_percentages.get(
                "Higher-Income Customers",
                0
            )


        lower_income_percentage = \
            segment_percentages.get(
                "Lower-Income Customers",
                0
            )


        # ==========================================
        # TOP METRICS
        # ==========================================

        st.subheader(
            "📊 Dataset Overview"
        )


        metric1, metric2, metric3, metric4 = \
            st.columns(4)


        with metric1:

            st.metric(
                "👥 Total Customers",
                f"{total_customers:,}"
            )


        with metric2:

            st.metric(
                "🔵 Higher-Income",
                f"{higher_income_percentage:.2f}%"
            )


        with metric3:

            st.metric(
                "🟢 Lower-Income",
                f"{lower_income_percentage:.2f}%"
            )


        with metric4:

            st.metric(
                "🔢 Number of Clusters",
                number_of_clusters
            )


        st.divider()


        # ==========================================
        # SEGMENT DISTRIBUTION
        # ==========================================

        st.subheader(
            "👥 Customer Segment Distribution"
        )


        segment_counts = (
            df["Segment_Name"]
            .value_counts()
        )


        chart_col1, chart_col2 = \
            st.columns(2)


        with chart_col1:

            st.bar_chart(
                segment_counts,
                height=350
            )


        with chart_col2:

            st.dataframe(
                pd.DataFrame({

                    "Customer Segment":
                        segment_counts.index,

                    "Customers":
                        segment_counts.values,

                    "Percentage":
                        [
                            round(
                                value /
                                total_customers *
                                100,
                                2
                            )
                            for value in
                            segment_counts.values
                        ]

                }),
                use_container_width=True,
                hide_index=True
            )


        st.divider()


        # ==========================================
        # PCA CLUSTER VISUALIZATION
        # ==========================================

        st.subheader(
            "🔬 K-Means Cluster Visualization"
        )


        st.write(
            "Two-dimensional PCA visualization of "
            "the customer clusters."
        )


        if model_loaded:

            try:

                preprocessor = \
                    model_data["preprocessor"]


                X_for_pca = df[
                    model_data["feature_columns"]
                ].copy()


                processed_data = \
                    preprocessor.transform(
                        X_for_pca
                    )


                pca_2d = PCA(
                    n_components=2,
                    random_state=42
                )


                X_pca_2d = \
                    pca_2d.fit_transform(
                        processed_data
                    )


                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )


                for cluster in sorted(
                    df["Cluster"].unique()
                ):

                    mask = (
                        df["Cluster"]
                        .values
                        == cluster
                    )


                    if cluster == 0:

                        label = \
                            "Higher-Income Customers"

                    else:

                        label = \
                            "Lower-Income Customers"


                    ax.scatter(
                        X_pca_2d[mask, 0],
                        X_pca_2d[mask, 1],
                        alpha=0.35,
                        s=10,
                        label=label
                    )


                ax.set_title(
                    "K-Means Customer Clusters"
                )

                ax.set_xlabel(
                    "Principal Component 1"
                )

                ax.set_ylabel(
                    "Principal Component 2"
                )

                ax.legend()

                ax.grid(
                    True,
                    alpha=0.2
                )


                st.pyplot(
                    fig,
                    use_container_width=True
                )


            except Exception as e:

                st.warning(
                    "PCA visualization could not be generated."
                )

                st.code(
                    str(e)
                )


        else:

            st.info(
                "Model file was not found."
            )


        st.divider()


        # ==========================================
        # FINANCIAL COMPARISON
        # ==========================================

        st.subheader(
            "💰 Financial Comparison"
        )


        financial_profile = (
            df.groupby("Segment_Name")[
                [
                    "Income Level",
                    "Coverage Amount",
                    "Premium Amount"
                ]
            ]
            .mean()
            .round(2)
        )


        st.dataframe(
            financial_profile,
            use_container_width=True
        )


        # ==========================================
        # CHARTS
        # ==========================================

        chart1, chart2 = \
            st.columns(2)


        with chart1:

            st.subheader(
                "💵 Average Income"
            )


            income_data = (
                df.groupby("Segment_Name")[
                    "Income Level"
                ]
                .mean()
                .round(2)
            )


            st.bar_chart(
                income_data,
                height=300
            )


        with chart2:

            st.subheader(
                "🎂 Average Age"
            )


            age_data = (
                df.groupby("Segment_Name")[
                    "Age"
                ]
                .mean()
                .round(2)
            )


            st.bar_chart(
                age_data,
                height=300
            )


        chart3, chart4 = \
            st.columns(2)


        with chart3:

            st.subheader(
                "🛡️ Average Coverage"
            )


            coverage_data = (
                df.groupby("Segment_Name")[
                    "Coverage Amount"
                ]
                .mean()
                .round(2)
            )


            st.bar_chart(
                coverage_data,
                height=300
            )


        with chart4:

            st.subheader(
                "💳 Average Premium"
            )


            premium_data = (
                df.groupby("Segment_Name")[
                    "Premium Amount"
                ]
                .mean()
                .round(2)
            )


            st.bar_chart(
                premium_data,
                height=300
            )


        st.divider()


        # ==========================================
        # SEGMENT PROFILE
        # ==========================================

        st.subheader(
            "🔎 Segment Profile"
        )


        segment_profile = (
            df.groupby("Segment_Name")
            .agg(

                Customer_Count=(
                    "Customer ID",
                    "count"
                ),

                Average_Age=(
                    "Age",
                    "mean"
                ),

                Average_Income=(
                    "Income Level",
                    "mean"
                ),

                Average_Coverage=(
                    "Coverage Amount",
                    "mean"
                ),

                Average_Premium=(
                    "Premium Amount",
                    "mean"
                )

            )
            .round(2)
        )


        st.dataframe(
            segment_profile,
            use_container_width=True
        )


        st.divider()


        # ==========================================
        # DOWNLOAD DATA
        # ==========================================

        st.subheader(
            "📥 Download Segmented Customer Data"
        )


        st.write(
            "Download the complete customer dataset "
            "with the generated cluster and segment name."
        )


        csv_data = (
            df.to_csv(
                index=False
            )
            .encode("utf-8")
        )


        st.download_button(
            label=
                "📥 Download Customer Segmentation CSV",

            data=csv_data,

            file_name=
                "customer_segmentation_results.csv",

            mime="text/csv",

            use_container_width=True
        )


# ==================================================
# PAGE 3 — ABOUT PROJECT
# ==================================================

elif page == "ℹ️ About Project":

    st.title(
        "ℹ️ About the Project"
    )


    st.subheader(
        "Intelligent Customer Segmentation "
        "and Behavioural Analytics"
    )


    st.write(
        """
        This project uses unsupervised machine learning
        to identify customer segments from demographic,
        financial and behavioural information.
        """
    )


    st.subheader(
        "🤖 Machine Learning"
    )


    st.write(
        """
        K-Means Clustering is used as the primary
        customer segmentation algorithm.

        Since this is an unsupervised learning problem,
        the existing Segmentation Group column is not
        used as a training target.
        """
    )


    st.subheader(
        "⚙️ Data Processing"
    )


    st.write(
        """
        The dataset goes through the following
        preprocessing steps:

        • Missing value handling

        • Numerical feature scaling

        • Categorical feature encoding

        • Principal Component Analysis (PCA)

        • K-Means clustering
        """
    )


    st.subheader(
        "🔬 Algorithms Compared"
    )


    st.write(
        """
        • K-Means Clustering

        • Hierarchical Clustering

        • DBSCAN
        """
    )


    st.subheader(
        "📊 Clustering Evaluation"
    )


    st.write(
        """
        • Silhouette Score

        • Davies-Bouldin Index

        • Calinski-Harabasz Index
        """
    )


    st.subheader(
        "🛠️ Technology Stack"
    )


    tech_col1, tech_col2 = \
        st.columns(2)


    with tech_col1:

        st.write(
            """
            **Machine Learning**

            • Python

            • Pandas

            • NumPy

            • Scikit-learn

            • K-Means

            • PCA
            """
        )


    with tech_col2:

        st.write(
            """
            **Application**

            • FastAPI

            • Streamlit

            • REST API

            • Joblib

            • Google Colab

            • Kaggle Dataset
            """
        )


    st.subheader(
        "🔄 Project Pipeline"
    )


    st.code(
        """
Dataset
   ↓
Data Preprocessing
   ↓
Feature Encoding
   ↓
Feature Scaling
   ↓
PCA
   ↓
K-Means Clustering
   ↓
Customer Segments
   ↓
Model (.pkl)
   ↓
FastAPI Backend
   ↓
Streamlit Frontend
   ↓
Prediction + Analytics
        """
    )


    st.success(
        "End-to-end customer segmentation system "
        "is successfully implemented."
    )