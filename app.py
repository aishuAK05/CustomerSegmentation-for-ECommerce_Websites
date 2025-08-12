import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the saved scaler
with open("ml.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load the processed RFM dataset
rfm = pd.read_csv("rfm_data.csv", index_col=0)

# Normalize the data
rfm_scaled = scaler.transform(rfm)

# Apply K-Means Clustering
num_clusters = 4  # Adjust if needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(rfm_scaled)

# Define customer categories based on clusters
customer_categories = {
    0: "Champions",
    1: "Loyal Customers",
    2: "Potential Loyalist",
    3: "At-Risk Customers",
    4: "Hibernating",
    5: "Promising Customers"
}
# Map clusters to category names
rfm["Customer_Segment"] = rfm["Cluster"].map(customer_categories)


# Streamlit UI
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("📊 Customer Segmentation using RFM & K-Means")
st.markdown(
    "This app segments customers based on **Recency, Frequency, and Monetary (RFM)** values using K-Means clustering."
)

# Sidebar for navigation
st.sidebar.header("Navigation")
selected_option = st.sidebar.radio("Select an option:", ["Overview", "Cluster Insights", "Download Data"])

if selected_option == "Overview":
    st.subheader("📌 Processed RFM Data with Clusters")
    st.write("Below is the processed RFM dataset with assigned cluster labels.")
    st.dataframe(rfm)

elif selected_option == "Visualizations":
    st.subheader("📊 Data Visualizations")
     # Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(rfm.drop(columns=["Customer_Segment", "Cluster"]).corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    st.pyplot(fig)

    # RFM Distribution
    st.write("### RFM Distribution by Segment")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=rfm["Customer_Segment"], y=rfm["Monetary"], palette="viridis")
    plt.xticks(rotation=45)
    plt.xlabel("Customer Segment")
    plt.ylabel("Monetary Value")
    plt.title("Monetary Value Distribution Across Segments")
    st.pyplot(fig)

    # Scatterplot
    st.write("### RFM Scatterplot")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=rfm["Recency"], y=rfm["Monetary"], hue=rfm["Customer_Segment"], palette="tab10", s=100)
    plt.xlabel("Recency (Days Since Last Purchase)")
    plt.ylabel("Monetary (Total Spending)")
    plt.title("Customer Segmentation Scatterplot")
    st.pyplot(fig)


    # Pairplot
    st.write("### RFM Pairplot (Colored by Cluster)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=rfm["Recency"], y=rfm["Monetary"], hue=rfm["Cluster"], palette="viridis", s=100)
    plt.xlabel("Recency (Days Since Last Purchase)")
    plt.ylabel("Monetary (Total Spending)")
    plt.title("Customer Segmentation Scatterplot")
    st.pyplot(fig)

elif selected_option == "Cluster Insights":
    st.subheader("🔍 Cluster Insights")
    
    # Cluster distribution
    st.write("### Cluster Distribution")
    cluster_counts = rfm["Customer_Segment"].value_counts()
    st.bar_chart(cluster_counts)
    
    # Cluster Summary
    st.write("### Cluster Characteristics")
    cluster_summary = rfm.groupby("Customer_Segment").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean"
    }).reset_index()
    st.dataframe(cluster_summary)   

elif selected_option == "Download Data":
    st.subheader("📥 Download Clustered Data")
    st.write("Click the button below to download the RFM dataset with cluster labels.")
    st.download_button(
        label="Download CSV",
        data=rfm.to_csv().encode("utf-8"),
        file_name="rfm_clustered.csv",
        mime="text/csv"
    )

st.sidebar.markdown("Developed with ❤️ using Streamlit")
