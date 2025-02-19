import streamlit as st
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime, timedelta

# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://prithvi:prithvi@clusterprithvi.c7lzckp.mongodb.net/?retryWrites=true&w=majority") 
db = client["youtube_analysis"]

# Add auto-refresh to the page (refresh every 10 seconds)
st.markdown('<meta http-equiv="refresh" content="120">', unsafe_allow_html=True)

# Dashboard Title
st.title("YouTube Streaming Trend Analysis Dashboard")

# Sidebar for Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Sliding Window Metrics", "Top Categories", "Top Hashtags", "Top Videos"])


# Fetch and Display Sliding Window Metrics
if options == "Sliding Window Metrics":
    st.header("Sliding Window Metrics")
    
    # Fetch the most recent document (sorted by `_id` descending)
    metrics = db["sliding_window_metrics"].find_one(sort=[("_id", -1)])  # `_id` auto-generated by MongoDB
    
    if metrics:
        # Styled metrics using Markdown
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                <div style="background-color: #2ca02c; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>Total Views</h3>
                    <h1>{metrics["total_views"]}</h1>
                </div>
                <div style="background-color: #ff7f0e; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>Total Likes</h3>
                    <h1>{metrics["total_likes"]}</h1>
                </div>
                <div style="background-color: #1f77b4; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>Total Comments</h3>
                    <h1>{metrics["total_comments"]}</h1>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("No data found for sliding window metrics.")


# Fetch and Display Top Categories
elif options == "Top Categories":
    st.header("Top Trending Categories")
    
    # Fetch the 10 most recent documents, sorted by _id descending
    recent_categories = list(db["top_categories"].find().sort("_id", -1).limit(10))
    
    # Convert to DataFrame
    categories = pd.DataFrame(recent_categories)
    
    if not categories.empty:
        # Drop the `_id` column
        categories = categories.drop(columns=["_id"])
        categories = categories.sort_values(by="total_views", ascending=False)
        st.dataframe(categories)
        # Bar Chart
        st.subheader("Category-wise Views")
        plt.figure(figsize=(10, 5))
        plt.bar(categories["category"], categories["total_views"], color="skyblue")
        plt.xlabel("Category")
        plt.ylabel("Total Views")
        plt.xticks(rotation=45)
        st.pyplot(plt)
    else:
        st.warning("No recent data found for top categories.")




# Fetch and Display Top Hashtags
elif options == "Top Hashtags":
    st.header("Top Trending Hashtags")

    # Fetch the 10 most recent documents, sorted by _id descending
    recent_hashtags = list(db["top_hashtags"].find().sort("_id", -1).limit(10))
    
    if recent_hashtags:
        # Combine all hashtags into a single list
        all_hashtags = [doc["hashtag"] for doc in recent_hashtags]
        
        # Create a Word Cloud
        st.subheader("Hashtag Word Cloud")
        wordcloud = WordCloud(
            background_color="white",
            width=800,
            height=400,
            colormap="viridis"
        ).generate(" ".join(all_hashtags))
        
        # Display the Word Cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.warning("No recent data found for top hashtags.")


# Fetch and Display Top Videos
elif options == "Top Videos":
    st.header("Top Trending Videos")
    
    # Fetch the 10 most recent documents for videos by likes
    st.subheader("By Likes")
    recent_videos_by_likes = pd.DataFrame(list(db["top_videos_by_likes"].find().sort("_id", -1).limit(10)))
    if not recent_videos_by_likes.empty:
        st.dataframe(recent_videos_by_likes)
    else:
        st.warning("No recent data found for top videos by likes.")
    
    # Fetch the 10 most recent documents for videos by views
    st.subheader("By Views")
    recent_videos_by_views = pd.DataFrame(list(db["top_videos_by_views"].find().sort("_id", -1).limit(10)))
    if not recent_videos_by_views.empty:
        st.dataframe(recent_videos_by_views)
    else:
        st.warning("No recent data found for top videos by views.")

