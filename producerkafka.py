from googleapiclient.discovery import build
from kafka import KafkaProducer
import json
import re
import time
import ssl
from bloom_filter2 import BloomFilter


# Disable SSL verification for Kafka
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

# Initialize YouTube API with your API key
API_KEY = ''
youtube = build("youtube", "v3", developerKey=API_KEY)

# Initialize Kafka Producer with your credentials
youTube_producer = KafkaProducer(
    bootstrap_servers='pkc-619z3.us-east1.gcp.confluent.cloud:9092',
    security_protocol="SASL_SSL",
    sasl_mechanism="PLAIN",
    sasl_plain_username='',
    sasl_plain_password='',
    ssl_context=context,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Initialize Bloom Filter for hashtags
hashtag_bloom_filter = BloomFilter(max_elements=10000, error_rate=0.001)

# Fetch categories and create a mapping for the US region
def get_categories(region_code='US'):
    """Fetch category mapping from YouTube API."""
    request = youtube.videoCategories().list(
        part="snippet",
        regionCode=region_code
    )
    response = request.execute()

    category_map = {}
    for item in response['items']:
        category_map[item['id']] = item['snippet']['title']
    return category_map

# Function to extract hashtags from descriptions
def extract_hashtags(description):
    """Extract hashtags from video description."""
    return re.findall(r"#(\w+)", description)

# Check if a hashtag is unique using the Bloom Filter
def is_unique_hashtag(hashtag):
    """Check if a hashtag is unique."""
    if hashtag in hashtag_bloom_filter:
        return False
    hashtag_bloom_filter.add(hashtag)
    return True

# Function to get video data
def get_video_data(video_id, categories):
    try:
        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()

        if len(response['items']) > 0:
            video_data = response['items'][0]
            snippet = video_data['snippet']
            statistics = video_data['statistics']

            # Extract video details
            category_id = snippet.get('categoryId', '')
            category_name = categories.get(category_id, 'Unknown')  # Map categoryId to category name

            # Extract hashtags and filter unique ones
            all_hashtags = extract_hashtags(snippet.get('description', ''))
            unique_hashtags = [hashtag for hashtag in all_hashtags if is_unique_hashtag(hashtag)]

            data = {
                'video_id': video_data['id'],
                'title': snippet['title'],
                'published_at': snippet['publishedAt'],
                'channel': snippet['channelTitle'],
                'description': snippet.get('description', ''),
                'hashtags': unique_hashtags,  
                'category': category_name,
                'views': int(statistics.get('viewCount', 0)),
                'likes': int(statistics.get('likeCount', 0)),
                'comments': int(statistics.get('commentCount', 0))
            }

            return data
    except Exception as e:
        print(f"Error fetching video data: {e}")
        return None

# Function to search and produce videos for the US region
def search_and_produce_videos(categories):
    """Search videos and send them to Kafka for the US region."""
    request = youtube.search().list(
        part="snippet",
        maxResults=150,
        order="date",
        type="video",
        relevanceLanguage="en",
        regionCode="US"
    )
    response = request.execute()

    for item in response['items']:
        video_id = item['id']['videoId']
        video_data = get_video_data(video_id, categories)

        if video_data:
            youTube_producer.send("youtube_topic", value=video_data)
            print(f"Produced: {video_data['title']} with category {video_data['category']} to Kafka")
            time.sleep(1)

# Main function
if __name__ == "__main__":
    categories = get_categories(region_code='US')
    while True:
        search_and_produce_videos(categories)
        time.sleep(60)  # Wait to avoid exceeding API limits
