import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="🧠 YAXΞ Customer Intelligence", layout="wide")

# =========================
# TITLE
# =========================
st.title("YAXΞ Customer Intelligence")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("segmentation.csv")

st.subheader("📊 Raw Data")
st.dataframe(df.head())

# =========================
# FEATURE SELECTION (PRO)
# =========================
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# =========================
# ELBOW METHOD
# =========================
st.subheader("📊 Optimal Clusters (Elbow Method)")

inertia = []
K = range(2, 10)

for k_val in K:
    kmeans_test = KMeans(n_clusters=k_val, random_state=42)
    kmeans_test.fit(X)
    inertia.append(kmeans_test.inertia_)

plt.figure()
plt.plot(K, inertia, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
st.pyplot(plt)

# =========================
# SIDEBAR CONTROL
# =========================
k = st.sidebar.slider("Select number of clusters", 2, 10, 5)

# =========================
# MODEL
# =========================
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# =========================
# METRICS
# =========================
st.subheader("📊 Cluster Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Avg Income", round(df["Annual Income (k$)"].mean(), 2))
col3.metric("Avg Spending Score", round(df["Spending Score (1-100)"].mean(), 2))

# =========================
# CLUSTER SUMMARY
# =========================
cluster_summary = df.groupby("Cluster").mean(numeric_only=True)

st.subheader("📈 Cluster Summary")
st.dataframe(cluster_summary)

# =========================
# CUSTOMER PERSONAS (KEY)
# =========================
st.subheader("🧠 Customer Personas")

for i in range(k):
    avg_income = cluster_summary.loc[i, "Annual Income (k$)"]
    avg_spending = cluster_summary.loc[i, "Spending Score (1-100)"]

    if avg_income > 70 and avg_spending > 70:
        persona = "💎 High Income – High Spenders"
        insight = "Premium customers. Retain with VIP perks."

    elif avg_income < 40 and avg_spending > 60:
        persona = "💸 Low Income – High Spenders"
        insight = "Risky group. Offer discounts and retention strategies."

    elif avg_income > 70 and avg_spending < 40:
        persona = "🧊 High Income – Low Spenders"
        insight = "Untapped segment. Target with personalized marketing."

    else:
        persona = "💤 Low Engagement Customers"
        insight = "Low priority. Use basic engagement campaigns."

    st.write(f"### Cluster {i}: {persona}")
    st.info(insight)

# =========================
# SEGMENT DISTRIBUTION
# =========================
st.subheader("📊 Customers per Segment")
st.bar_chart(df["Cluster"].value_counts())

# =========================
# 🔥 PLOTLY VISUALIZATION
# =========================
st.subheader("📉 Customer Segmentation Visualization (Interactive)")

fig = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color=df["Cluster"].astype(str),
    hover_data=["Age", "Gender"],
    title="Customer Segments (Interactive View)"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# DOWNLOAD
# =========================
st.subheader("⬇️ Download Results")

csv = df.to_csv(index=False)

st.download_button(
    label="Download Clustered Data",
    data=csv,
    file_name="customer_segments.csv",
    mime="text/csv"
)
