import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation & Retention",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }
    
    .main { background-color: #0d1117; color: #e6edf3; }
    
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 4px;
    }
    .segment-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px;
    }
    .champion { background: #1f4e2e; color: #3fb950; border: 1px solid #3fb950; }
    .loyal { background: #1a3a5c; color: #58a6ff; border: 1px solid #58a6ff; }
    .atrisk { background: #4a2d00; color: #f0883e; border: 1px solid #f0883e; }
    .lost { background: #3d1a1a; color: #f85149; border: 1px solid #f85149; }
    
    .stButton>button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        width: 100%;
    }
    .prediction-box {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
    }
    .churned { background: linear-gradient(135deg, #3d1a1a, #5a1e1e); border: 2px solid #f85149; }
    .active { background: linear-gradient(135deg, #1f4e2e, #1a3d28); border: 2px solid #3fb950; }
    
    div[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stSelectbox label, .stNumberInput label, .stSlider label { color: #8b949e !important; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading & Processing ─────────────────────────────────────────────────
@st.cache_data
def load_and_process(file):
    data = pd.read_csv(file)
    data = data.dropna(subset=["CustomerID"])
    data = data[~data["InvoiceNo"].astype(str).str.startswith("C")]
    data = data[data["Quantity"] > 0]
    data = data[data["UnitPrice"] > 0]
    data["Description"] = data["Description"].fillna(data["StockCode"])
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["CustomerID"] = data["CustomerID"].astype("int64")

    reference_date = data["InvoiceDate"].max()

    # Customer level
    df = data.groupby("CustomerID")["InvoiceDate"].max().reset_index()
    df.columns = ["CustomerID", "LastPurchaseDate"]
    df["recency"] = (reference_date - df["LastPurchaseDate"]).dt.days
    df["churn_status"] = (df["recency"] > 90).astype(int)

    freq = data.groupby("CustomerID")["InvoiceNo"].nunique().reset_index(name="Frequency")
    df = pd.merge(df, freq, on="CustomerID")

    data["Revenue"] = data["Quantity"] * data["UnitPrice"]
    cv = data.groupby("CustomerID")["Revenue"].sum().reset_index(name="Customer_value")
    df = pd.merge(df, cv, on="CustomerID")

    first = data.groupby("CustomerID")["InvoiceDate"].min().reset_index(name="FirstPurchaseDate")
    df = pd.merge(df, first, on="CustomerID")
    df["CustomerLifespan"] = (df["LastPurchaseDate"] - df["FirstPurchaseDate"]).dt.days
    df["customer_lifetime_value"] = df["Customer_value"] * df["CustomerLifespan"]
    df = df.drop(columns=["LastPurchaseDate", "FirstPurchaseDate"])

    df["R_score"] = pd.qcut(df["recency"], q=4, labels=[4,3,2,1]).astype(int)
    df["F_score"] = pd.cut(df["Frequency"], bins=[0,1,3,6,float("inf")], labels=[1,2,3,4]).astype(int)
    df["M_score"] = pd.qcut(df["Customer_value"], q=4, labels=[1,2,3,4]).astype(int)
    df["RFM_score"] = df["R_score"] + df["F_score"] + df["M_score"]

    def segment(score):
        if score >= 10: return "Champion"
        elif score >= 7: return "Loyal"
        elif score >= 5: return "At Risk"
        else: return "Lost"
    df["Segment"] = df["RFM_score"].apply(segment)

    return df, data


@st.cache_resource
def train_model(df):
    features = ["Frequency", "Customer_value", "CustomerLifespan", "customer_lifetime_value"]
    X = df[features]
    y = df["churn_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return model, acc, report, cm, importance


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Upload Dataset")
    uploaded = st.file_uploader("Upload Online Retail CSV", type=["csv"])
    st.markdown("---")
    st.markdown("## 🔍 Filters")

    if uploaded:
        df, raw = load_and_process(uploaded)
        segment_filter = st.multiselect(
            "Customer Segment",
            options=df["Segment"].unique().tolist(),
            default=df["Segment"].unique().tolist()
        )
        churn_filter = st.selectbox("Churn Status", ["All", "Active (0)", "Churned (1)"])
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.markdown(f"**Transactions:** {len(raw):,}")
        st.markdown(f"**Customers:** {len(df):,}")
        st.markdown(f"**Countries:** {raw['Country'].nunique()}")
    else:
        st.info("Upload your CSV file to begin")


# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown("# 📊 Customer Segmentation & Retention")
st.markdown("##### *Online Retail — RFM Analysis · CLV · Churn Prediction*")
st.markdown("---")

if not uploaded:
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px; color: #8b949e;'>
        <div style='font-size: 4rem;'>📂</div>
        <div style='font-family: Syne; font-size: 1.5rem; color: #e6edf3; margin: 16px 0;'>Upload your dataset to get started</div>
        <div>Upload the Online Retail CSV file using the sidebar to explore customer insights</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Apply filters
filtered = df[df["Segment"].isin(segment_filter)]
if churn_filter == "Active (0)":
    filtered = filtered[filtered["churn_status"] == 0]
elif churn_filter == "Churned (1)":
    filtered = filtered[filtered["churn_status"] == 1]

# Train model
model, acc, report, cm, importance = train_model(df)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "👥 RFM Segments", "🤖 Churn Model", "🔮 Predict"])

# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    churn_rate = df["churn_status"].mean() * 100
    avg_clv = df["customer_lifetime_value"].median()
    avg_freq = df["Frequency"].mean()
    avg_spend = df["Customer_value"].mean()

    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(df):,}</div>
            <div class='metric-label'>Total Customers</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{churn_rate:.1f}%</div>
            <div class='metric-label'>Churn Rate</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{avg_freq:.1f}</div>
            <div class='metric-label'>Avg Purchase Frequency</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>£{avg_spend:,.0f}</div>
            <div class='metric-label'>Avg Customer Spend</div></div>""", unsafe_allow_html=True)

    st.markdown("#### ")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Recency Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        ax.hist(df["recency"], bins=40, color='#58a6ff', alpha=0.8, edgecolor='#0d1117')
        ax.axvline(90, color='#f85149', linestyle='--', linewidth=1.5, label='Churn threshold (90d)')
        ax.set_xlabel("Days Since Last Purchase", color='#8b949e')
        ax.set_ylabel("Customers", color='#8b949e')
        ax.tick_params(colors='#8b949e')
        ax.legend(facecolor='#1c2333', labelcolor='#e6edf3', fontsize=8)
        for spine in ax.spines.values(): spine.set_color('#30363d')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("##### CLV Distribution (Top 95%)")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        clv_data = df[df["customer_lifetime_value"] < df["customer_lifetime_value"].quantile(0.95)]["customer_lifetime_value"]
        ax.hist(clv_data, bins=40, color='#3fb950', alpha=0.8, edgecolor='#0d1117')
        ax.set_xlabel("Customer Lifetime Value", color='#8b949e')
        ax.set_ylabel("Customers", color='#8b949e')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values(): spine.set_color('#30363d')
        st.pyplot(fig)
        plt.close()

    st.markdown("##### Sample Customer Data")
    display_cols = ["CustomerID", "recency", "churn_status", "Frequency", "Customer_value", "CustomerLifespan", "customer_lifetime_value", "RFM_score", "Segment"]
    st.dataframe(
        filtered[display_cols].head(100).style.background_gradient(subset=["RFM_score"], cmap="Blues"),
        use_container_width=True, height=300
    )


# ── Tab 2: RFM Segments ────────────────────────────────────────────────────────
with tab2:
    seg_counts = df["Segment"].value_counts()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Segment Distribution")
        colors = {"Champion": "#3fb950", "Loyal": "#58a6ff", "At Risk": "#f0883e", "Lost": "#f85149"}
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        wedge_colors = [colors[s] for s in seg_counts.index]
        wedges, texts, autotexts = ax.pie(
            seg_counts.values, labels=seg_counts.index,
            colors=wedge_colors, autopct='%1.1f%%',
            textprops={'color': '#e6edf3', 'fontsize': 10},
            wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2}
        )
        for at in autotexts: at.set_color('#0d1117'); at.set_fontweight('bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("##### Avg Metrics by Segment")
        seg_summary = df.groupby("Segment").agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Spend=("Customer_value", "mean"),
            Churn_Rate=("churn_status", "mean")
        ).round(1)
        seg_summary["Churn_Rate"] = (seg_summary["Churn_Rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(seg_summary, use_container_width=True)

        st.markdown("##### Segment Legend")
        st.markdown("""
        <span class='segment-badge champion'>🏆 Champion: Score 10-12</span>
        <span class='segment-badge loyal'>⭐ Loyal: Score 7-9</span>
        <span class='segment-badge atrisk'>⚠️ At Risk: Score 5-6</span>
        <span class='segment-badge lost'>❌ Lost: Score 3-4</span>
        """, unsafe_allow_html=True)

    st.markdown("##### RFM Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    rfm_counts = df["RFM_score"].value_counts().sort_index()
    bar_colors = []
    for score in rfm_counts.index:
        if score >= 10: bar_colors.append('#3fb950')
        elif score >= 7: bar_colors.append('#58a6ff')
        elif score >= 5: bar_colors.append('#f0883e')
        else: bar_colors.append('#f85149')
    ax.bar(rfm_counts.index, rfm_counts.values, color=bar_colors, edgecolor='#0d1117', linewidth=0.5)
    ax.set_xlabel("RFM Score", color='#8b949e')
    ax.set_ylabel("Number of Customers", color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values(): spine.set_color('#30363d')
    st.pyplot(fig)
    plt.close()


# ── Tab 3: Churn Model ─────────────────────────────────────────────────────────
with tab3:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{acc*100:.1f}%</div>
            <div class='metric-label'>Model Accuracy</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{report['1']['recall']*100:.1f}%</div>
            <div class='metric-label'>Churned Customer Recall</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{report['1']['f1-score']*100:.1f}%</div>
            <div class='metric-label'>Churned F1-Score</div></div>""", unsafe_allow_html=True)

    st.markdown("#### ")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Active', 'Churned'],
                    yticklabels=['Active', 'Churned'],
                    ax=ax, linewidths=2, linecolor='#0d1117')
        ax.set_xlabel("Predicted", color='#8b949e')
        ax.set_ylabel("Actual", color='#8b949e')
        ax.tick_params(colors='#8b949e')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("##### Feature Importance")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        colors_imp = ['#58a6ff', '#3fb950', '#f0883e', '#f85149']
        bars = ax.barh(importance.index[::-1], importance.values[::-1],
                       color=colors_imp[::-1], edgecolor='#0d1117')
        ax.set_xlabel("Importance Score", color='#8b949e')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values(): spine.set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("##### Classification Report")
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)


# ── Tab 4: Predict ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("##### 🔮 Predict Customer Churn")
    st.markdown("Enter customer behavior details below to get an instant churn prediction.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        freq_input = st.number_input("Number of Purchases (Frequency)", min_value=1, max_value=500, value=3)
        spend_input = st.number_input("Total Amount Spent (£)", min_value=0.0, max_value=100000.0, value=500.0, step=50.0)
    with col2:
        lifespan_input = st.number_input("Customer Lifespan (days)", min_value=0, max_value=1000, value=180)
        clv_input = spend_input * lifespan_input

    st.markdown(f"*Calculated CLV: **£{clv_input:,.0f}***")
    st.markdown("---")

    if st.button("🔮 Predict Churn"):
        input_data = pd.DataFrame([[freq_input, spend_input, lifespan_input, clv_input]],
                                   columns=["Frequency", "Customer_value", "CustomerLifespan", "customer_lifetime_value"])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.markdown(f"""
            <div class='prediction-box churned'>
                <div style='font-size:3rem'>🔴</div>
                <div style='font-family:Syne; font-size:1.8rem; color:#f85149; font-weight:800; margin:8px 0'>LIKELY TO CHURN</div>
                <div style='color:#8b949e'>Churn Probability: <strong style='color:#f85149'>{probability[1]*100:.1f}%</strong></div>
                <div style='color:#8b949e; margin-top:8px; font-size:0.9rem'>⚠️ Recommend immediate retention action — offer discount or personalized outreach</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='prediction-box active'>
                <div style='font-size:3rem'>🟢</div>
                <div style='font-family:Syne; font-size:1.8rem; color:#3fb950; font-weight:800; margin:8px 0'>LIKELY ACTIVE</div>
                <div style='color:#8b949e'>Active Probability: <strong style='color:#3fb950'>{probability[0]*100:.1f}%</strong></div>
                <div style='color:#8b949e; margin-top:8px; font-size:0.9rem'>✅ Customer appears engaged — consider loyalty rewards to maintain relationship</div>
            </div>""", unsafe_allow_html=True)

