import os
import streamlit as st
import sqlalchemy
import pandas as pd
import plotly.express as px

# DB connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://affan:pass123@postgres:5432/news_db")

# Category mapping
CATEGORY_MAP = {
    1: "Entertainment",
    2: "Sports",
    3: "International",
    4: "Business",
    5: "Technology"
}

# Try to connect
try:
    engine = sqlalchemy.create_engine(DATABASE_URL)

    @st.cache_data
    def load_data(limit=100):
        query = "SELECT * FROM predictions ORDER BY id DESC LIMIT %s"
        return pd.read_sql(query, engine, params=(limit,))

    df = load_data()
    df["category"] = df["predicted_label"].astype(int).map(CATEGORY_MAP)

except Exception as e:
    st.error(f"Database connection failed: {e}")
    df = pd.DataFrame()

# Layout
st.title("📰 Urdu News Predictions Dashboard")
st.markdown("Explore recent machine learning predictions for Urdu news articles.")

# Filters
st.sidebar.header("Filters")
if not df.empty:
    categories = st.sidebar.multiselect("Select Categories", df["category"].unique(), default=df["category"].unique())
    sentiments = st.sidebar.multiselect("Select Sentiments", df["predicted_sentiment"].unique(), default=df["predicted_sentiment"].unique())

    filtered_df = df[
        (df["category"].isin(categories)) &
        (df["predicted_sentiment"].isin(sentiments))
    ]
    st.sidebar.write(f"Showing {len(filtered_df)} articles.")
else:
    filtered_df = df
    st.sidebar.info("No data available for filtering.")

# Recent predictions
st.header("Recent Articles")
if not filtered_df.empty:
    st.dataframe(filtered_df[["id", "title", "category", "predicted_sentiment"]].head(10))
else:
    st.info("No data available.")

# Category Distribution
st.header("📊 Category Distribution")
if not filtered_df.empty:
    cat_counts = filtered_df["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig_bar = px.bar(cat_counts, x="Category", y="Count", color="Category", title="Articles by Category")
    st.plotly_chart(fig_bar)
else:
    st.info("No category data to display.")

# Sentiment Pie Chart
st.header("😊 Sentiment Distribution")
if not filtered_df.empty:
    sent_counts = filtered_df["predicted_sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]
    fig_pie = px.pie(sent_counts, names="Sentiment", values="Count", title="Sentiment Breakdown")
    st.plotly_chart(fig_pie)
else:
    st.info("No sentiment data to display.")

# Sentiment by Category
st.header("📊 Sentiment by Category")
if not filtered_df.empty:
    grouped = filtered_df.groupby(["category", "predicted_sentiment"]).size().reset_index(name="count")
    fig_grouped = px.bar(grouped, x="category", y="count", color="predicted_sentiment", barmode="stack", title="Sentiment Distribution by Category")
    st.plotly_chart(fig_grouped)
else:
    st.info("No sentiment-category data available.")

# Summary
st.header("📋 Summary")
if not filtered_df.empty:
    st.write("**Total Articles**:", len(filtered_df))
    st.write("**Category Breakdown**:", filtered_df["category"].value_counts().to_dict())
    st.write("**Sentiment Breakdown**:", filtered_df["predicted_sentiment"].value_counts().to_dict())
else:
    st.info("No summary available.")

