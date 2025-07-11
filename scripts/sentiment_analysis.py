import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Load raw review data
csv_path = 'data/raw_reviews.csv'
df = pd.read_csv(csv_path, encoding='ISO-8859-1')
df = df.dropna(subset=['reviews.text'])

# Calculate sentiment scores
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment_score'] = df['reviews.text'].apply(get_sentiment)

# Assign sentiment labels
def get_sentiment_label(score):
    if score > 0.2:
        return 'Positive'
    elif score < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)

# Fix review dates
df['reviews.date'] = pd.to_datetime(df['reviews.date'], errors='coerce')

# Save final CSV
os.makedirs('data', exist_ok=True)
df.to_csv('data/final_reviews.csv', index=False)

print("✅ Sentiment analysis complete!")

# ----------------- 🔍 ADVANCED ANALYSIS -----------------

# 1. Sentiment vs Rating Mismatch
def detect_mismatch(row):
    if row['reviews.rating'] >= 4 and row['sentiment_score'] < 0:
        return 'Positive Rating, Negative Text'
    elif row['reviews.rating'] <= 2 and row['sentiment_score'] > 0.2:
        return 'Negative Rating, Positive Text'
    else:
        return None

df['mismatch_flag'] = df.apply(detect_mismatch, axis=1)
df_mismatch = df[df['mismatch_flag'].notnull()]
df_mismatch.to_csv('data/mismatch_reviews.csv', index=False)

# 2. Top Positive & Negative Reviews
top_positive = df[df['sentiment_label'] == 'Positive'].sort_values(by='sentiment_score', ascending=False).head(3)
top_negative = df[df['sentiment_label'] == 'Negative'].sort_values(by='sentiment_score').head(3)

top_positive.to_csv('data/top_positive_reviews.csv', index=False)
top_negative.to_csv('data/top_negative_reviews.csv', index=False)

# 3. Category-wise Sentiment Summary
category_summary = df.groupby('categories')['sentiment_score'].mean().sort_values(ascending=False)
category_summary.to_csv('data/category_sentiment_summary.csv')

# 4. Save Summary Counts
summary_counts = df['sentiment_label'].value_counts()
summary_counts.to_csv('data/sentiment_counts.csv')

# 5. Visualizations
# Sentiment Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='sentiment_label', palette='Set2')
plt.title("Sentiment Distribution")
plt.tight_layout()
plt.savefig("data/sentiment_distribution.png")
plt.close()

# WordCloud for Positive
positive_text = " ".join(df[df['sentiment_label'] == 'Positive']['reviews.text'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud - Positive Reviews")
plt.tight_layout()
plt.savefig("data/wordcloud_positive.png")
plt.close()

# Sentiment Over Time
monthly_sentiment = df.groupby(df['reviews.date'].dt.to_period('M'))['sentiment_score'].mean()
plt.figure(figsize=(12, 6))
monthly_sentiment.plot(kind='line', marker='o')
plt.title("Average Sentiment Over Time")
plt.xlabel("Month")
plt.ylabel("Avg Sentiment Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/sentiment_over_time.png")
plt.close()

print("📊 Advanced analysis complete. Files saved to /data/")
