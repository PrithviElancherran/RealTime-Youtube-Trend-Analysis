from pymongo import MongoClient
import ssl
from kafka import KafkaConsumer
import json
from pyspark.sql.types import ArrayType, StringType
import numpy as np
import time
from pyspark.sql.functions import col, explode, lit, concat, collect_list, when, udf, desc, collect_set, sum as _sum
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, BucketedRandomProjectionLSH, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import pandas as pd
import shap
from pyspark.ml.linalg import DenseVector
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import certifi

# MongoDB Connection
client = MongoClient("mongodb+srv://prithvi:prithvi@clusterprithvi.c7lzckp.mongodb.net/?retryWrites=true&w=majority", tlsCAFile=certifi.where())  # Replace with your MongoDB URI
db = client["youtube_analysis"]  # Replace with your database name
collection = db["analysis_results"]

# Initialize Spark session
spark = SparkSession.builder \
    .appName("YouTubeTrendingAnalysis") \
    .getOrCreate()

# Kafka Consumer configuration
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

youTube_consumer = KafkaConsumer(
    'youtube_topic',  # Kafka topic
    bootstrap_servers='pkc-619z3.us-east1.gcp.confluent.cloud:9092',
    security_protocol="SASL_SSL",
    sasl_mechanism="PLAIN",
    sasl_plain_username='',
    sasl_plain_password='',
    ssl_context=ssl_context,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

@udf(returnType=ArrayType(StringType()))
def handle_empty_hashtags(hashtags):
    # Check for None or empty list and return ["none"] if true
    if hashtags is None or len(hashtags) == 0:
        return ["none"]
    return hashtags

# Differential Privacy Noise Function
def add_differential_privacy(value):
    noise = np.random.laplace(0, 1)  # Laplace noise for differential privacy
    return max(0, value + noise)  # Ensure non-negative values

# DGIM Implementation for sliding window counts
class DGIM:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buckets = []

    def add(self, value):
        current_time = int(time.time())
        self.buckets.append((value, current_time))
        self.buckets = [(v, t) for v, t in self.buckets if t > current_time - self.window_size]
        self._merge_buckets()

    def _merge_buckets(self):
        max_buckets = 2
        merged_buckets = []
        i = 0
        while i < len(self.buckets):
            if i + 1 < len(self.buckets) and self.buckets[i][0] == self.buckets[i + 1][0]:
                merged_buckets.append((self.buckets[i][0] * 2, self.buckets[i][1]))
                i += 2
            else:
                merged_buckets.append(self.buckets[i])
                i += 1
        self.buckets = merged_buckets

    def count(self):
        if not self.buckets:
            return 0
        count = sum(value for value, _ in self.buckets[:-1])
        count += self.buckets[-1][0] // 2
        return count

# Initialize DGIM instances for sliding window metrics
dgim_views = DGIM(window_size=3600)  # 1-hour sliding window
dgim_likes = DGIM(window_size=3600)
dgim_comments = DGIM(window_size=3600)

# Function to Save Results to MongoDB
def save_to_mongo(data, collection_name):
    """Save data to MongoDB collection."""
    try:
        collection = db[collection_name]
        if isinstance(data, list):
            collection.insert_many(data)
        else:
            collection.insert_one(data)
        print(f"Data successfully saved to MongoDB collection: {collection_name}")
    except Exception as e:
        print(f"Error saving data to MongoDB: {e}")

# Process and analyze video data
def process_data(video_data_list):
    df = spark.createDataFrame(video_data_list)

    # Apply differential privacy to views, likes, and comments
    df = df.withColumn("noisy_views", col("views") + np.random.laplace(0, 1))
    df = df.withColumn("noisy_likes", col("likes") + np.random.laplace(0, 1))
    df = df.withColumn("noisy_comments", col("comments") + np.random.laplace(0, 1))

    # **Sliding Window Metrics (DGIM)**
    print("\n--- Sliding Window Metrics (Last 1 Hour) ---")
    for row in df.collect():
        dgim_views.add(row["views"])
        dgim_likes.add(row["likes"])
        dgim_comments.add(row["comments"])

    sliding_metrics = {
            "total_views": dgim_views.count(),
            "total_likes": dgim_likes.count(),
            "total_comments": dgim_comments.count()
        }

    print(f"Total Views in Sliding Window: {dgim_views.count()}")
    print(f"Total Likes in Sliding Window: {dgim_likes.count()}")
    print(f"Total Comments in Sliding Window: {dgim_comments.count()}")

    save_to_mongo(sliding_metrics, "sliding_window_metrics")

    # **Top 10 Trending Videos**
    print("\n--- Top 10 Trending Videos by Likes ---")
    top_videos_by_likes = df.select("title", "category", "likes", "views", "comments").orderBy(desc("likes")).limit(10)
    top_videos_by_likes.show(truncate=False)
    save_to_mongo(top_videos_by_likes.toPandas().to_dict("records"), "top_videos_by_likes")

    print("\n--- Top 10 Trending Videos by Views ---")
    top_videos_by_views = df.select("title", "category", "likes", "views", "comments").orderBy(desc("views")).limit(10)
    top_videos_by_views.show(truncate=False)
    save_to_mongo(top_videos_by_views.toPandas().to_dict("records"), "top_videos_by_views")

    # **Top Trending Categories**
    print("\n--- Top Trending Categories ---")
    category_trends = df.groupBy("category").agg(
        _sum("views").alias("total_views")
    ).orderBy(desc("total_views"))
    category_trends.show(10, truncate=False)
    save_to_mongo(category_trends.toPandas().to_dict("records"), "top_categories")

    # **Top 10 Trending Hashtags**
    print("\n--- Top 10 Trending Hashtags ---")

    # Explode hashtags into rows and filter for English hashtags using a regex for ASCII characters
    hashtags_df = df.select(explode(col("hashtags")).alias("hashtag"))
    english_hashtags_df = hashtags_df.filter(hashtags_df["hashtag"].rlike("^[a-zA-Z0-9_]+$"))

    # Select only the hashtag column
    top_hashtags = english_hashtags_df.select("hashtag").limit(10)

    # Display the hashtags
    top_hashtags.show(truncate=False)

    # Save the hashtags to MongoDB
    save_to_mongo(top_hashtags.toPandas().to_dict("records"), "top_hashtags")

    # **Additional Step: Hashtag-Based Clustering**
    print("\n--- Hashtag-Based Clustering ---")

    # Handle empty or missing hashtags
    df = df.withColumn("hashtags", handle_empty_hashtags(col("hashtags")))

    # Explode hashtags into rows
    hashtags_df = df.select("video_id", "title", "hashtags").withColumn("hashtag", explode(col("hashtags")))

    # Convert hashtags into numerical vectors using TF-IDF
    tokenizer = Tokenizer(inputCol="hashtag", outputCol="tokens")
    tokenized_df = tokenizer.transform(hashtags_df)

    hashing_tf = HashingTF(inputCol="tokens", outputCol="raw_features", numFeatures=100)
    featurized_data = hashing_tf.transform(tokenized_df)

    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    idf_model = idf.fit(featurized_data)
    tfidf_df = idf_model.transform(featurized_data)

    # Apply LSH for clustering with explicitly set bucketLength
    lsh = BucketedRandomProjectionLSH(
        inputCol="tfidf_features",
        outputCol="hashes",
        numHashTables=3,
        bucketLength=2.0
    )
    lsh_model = lsh.fit(tfidf_df)
    clustered_df = lsh_model.transform(tfidf_df)

    # Approximate nearest neighbors (used for cluster assignment)
    cluster_assignments = lsh_model.approxSimilarityJoin(clustered_df, clustered_df, threshold=1.0, distCol="distance") \
        .select(col("datasetA.video_id").alias("video_a"),
                col("datasetB.video_id").alias("video_b"),
                col("distance"))

    print("\n--- Cluster Assignments ---")
    cluster_assignments.show(10, truncate=False)

    # Display some clustered data for verification
    print("\n--- Sample Clustered Hashtags ---")
    clustered_df.select("video_id", "title", "hashtag", "hashes").show(10, truncate=False)

    # Step 5: Assign cluster IDs based on hash similarity
    cluster_assignments = cluster_assignments.withColumn(
        "cluster_id",
        when(col("distance") <= 1.0, concat(col("video_a"), lit("_cluster")))
    )

    # Step 6: Join cluster assignments back with the original data
    clustered_with_ids = clustered_df.join(
        cluster_assignments,
        clustered_df["video_id"] == cluster_assignments["video_a"],
        "inner"
    ).select("cluster_id", "video_id", "title", "hashtag")

    clustered_with_ids = clustered_with_ids.withColumn(
        "cluster_id",
        when(col("hashtag") == "none", "none_cluster").otherwise(col("cluster_id"))
    )

    clustered_with_ids = clustered_with_ids.dropDuplicates(["cluster_id", "hashtag"])

    # Step 7: Group by cluster_id to collect unique hashtags in each cluster
    cluster_wise_hashtags = clustered_with_ids.groupBy("cluster_id") \
        .agg(collect_set("hashtag").alias("hashtags_in_cluster"))

    # Step 8: Display cluster-wise hashtags
    print("\n--- Cluster-Wise Hashtags (Unique) ---")
    cluster_wise_hashtags.show(truncate=False)

    # SHAP Explainability
    print("\n--- SHAP Explainability ---")

    vector_assembler = VectorAssembler(inputCols=["tfidf_features"], outputCol="features")
    classified_df = vector_assembler.transform(clustered_df)

    # Convert Spark DataFrame to NumPy array
    sklearn_features = np.array([row.features.toArray() for row in classified_df.collect()])
    sklearn_labels = np.random.randint(0, 2, sklearn_features.shape[0])

    sklearn_model = SklearnLogisticRegression(max_iter=1000)
    sklearn_model.fit(sklearn_features, sklearn_labels)

    # Create SHAP explainer
    explainer = shap.LinearExplainer(sklearn_model, sklearn_features)
    #explainer = shap.KernelExplainer(lsh_model.predict, train_summary)
    shap_values = explainer(sklearn_features)

    # Adjust SHAP Summary Plot
    print("Generating SHAP Summary Plot...")
    shap.summary_plot(
        shap_values,
        sklearn_features,
        feature_names=[f"feature_{i}" for i in range(sklearn_features.shape[1])],
        show=False
    )
    plt.title("SHAP Summary Plot (Adjusted Scale)")
    plt.savefig("shap_summary_plot_scaled.png")  # Save plot to file
    plt.show()

    # Adjust SHAP Force Plot using shap.Explanation
    print("Generating SHAP Force Plot...")
    feature_names = [f"feature_{i}" for i in range(sklearn_features.shape[1])]
    shap_explanation = shap.Explanation(
        values=shap_values,  # SHAP values
        base_values=explainer.expected_value,  # Expected value
        data=sklearn_features,  # Original feature data
        feature_names=feature_names  # Feature names
    )

    shap.force_plot(
        base_value=shap_explanation.base_values[0],
        shap_values=shap_explanation.values[0],
        features=shap_explanation.data[0],
        feature_names=feature_names,
        matplotlib=True,  # Use Matplotlib rendering
        show=False
    )
    plt.title("SHAP Force Plot (Adjusted Scale)")
    plt.savefig("shap_force_plot_scaled.png")  # Save plot to file
    plt.show()

    print("SHAP plots saved as 'shap_summary_plot_scaled.png' and 'shap_force_plot_scaled.png'.")



# Main function to consume and process data
if __name__ == "__main__":
    video_data_list = []
    print("Listening for messages on 'youtube_topic'...")
    for message in youTube_consumer:
        video_data = message.value
        video_data_list.append(video_data)
        if len(video_data_list) >= 25:
            process_data(video_data_list)
            video_data_list.clear()



