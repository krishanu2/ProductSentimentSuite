import pandas as pd
import os

def safe_read_csv(path, **kwargs):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
            return pd.DataFrame()
    else:
        print(f"⚠️ File not found: {path}")
        return pd.DataFrame()

# Load datasets
df = safe_read_csv('data/final_reviews.csv')
sentiment_counts = safe_read_csv('data/sentiment_counts.csv')
category_sentiment = safe_read_csv('data/category_sentiment_summary.csv')
top_pos = safe_read_csv('data/top_positive_reviews.csv')
top_neg = safe_read_csv('data/top_negative_reviews.csv')
mismatches = safe_read_csv('data/mismatch_reviews.csv')

# Format columns properly
if not sentiment_counts.empty:
    sentiment_counts.columns = ['sentiment_label', 'count']
    sentiment_counts['count'] = pd.to_numeric(sentiment_counts['count'], errors='coerce')

if not category_sentiment.empty:
    category_sentiment.columns = ['categories', 'avg_sentiment']
    category_sentiment['avg_sentiment'] = pd.to_numeric(category_sentiment['avg_sentiment'], errors='coerce')

# Begin report
report = []

report.append("📊 PRODUCT REVIEW SENTIMENT ANALYSIS REPORT")
report.append("=" * 60 + "\n")

# 1. Sentiment Distribution
report.append("1️⃣ SENTIMENT DISTRIBUTION:\n")
for _, row in sentiment_counts.iterrows():
    if pd.notnull(row['sentiment_label']) and pd.notnull(row['count']):
        report.append(f"- {row['sentiment_label']}: {int(row['count'])} reviews")
report.append("")

# 2. Most Problematic Category
report.append("2️⃣ MOST PROBLEMATIC CATEGORY:\n")
if not category_sentiment.empty:
    row = category_sentiment.sort_values(by='avg_sentiment').iloc[0]
    report.append(f"- Category: {row['categories']}")
    report.append(f"- Avg Sentiment Score: {float(row['avg_sentiment']):.2f}")
else:
    report.append("⚠️ No category data available.")
report.append("")

# 3. Most Loved & Most Criticized Products
report.append("3️⃣ PRODUCT POPULARITY BREAKDOWN:\n")
if not df.empty:
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
    product_sentiment = df.groupby('name')['sentiment_score'].mean().reset_index()
    most_loved = product_sentiment.sort_values(by='sentiment_score', ascending=False).head(1)
    most_hated = product_sentiment.sort_values(by='sentiment_score').head(1)

    if not most_loved.empty:
        report.append(f"🟢 Most Loved: {most_loved.iloc[0]['name']} ({most_loved.iloc[0]['sentiment_score']:.2f})")
    if not most_hated.empty:
        report.append(f"🔴 Most Criticized: {most_hated.iloc[0]['name']} ({most_hated.iloc[0]['sentiment_score']:.2f})")
else:
    report.append("⚠️ No product data.")
report.append("")

# 4. Suspicious Review Example
report.append("4️⃣ RATING vs. SENTIMENT MISMATCH:\n")
if not mismatches.empty:
    r = mismatches.iloc[0]
    report.append(f"- Username: {r['reviews.username']}")
    report.append(f"- Rating: {r['reviews.rating']}")
    report.append(f"- Sentiment: {float(r['sentiment_score']):.2f}")
    report.append(f"- Review: {r['reviews.text'][:150]}...")
else:
    report.append("✅ No mismatches found.")
report.append("")

# 5. Top Highlight Reviews
report.append("5️⃣ HIGHLIGHT REVIEWS:\n")
if not top_pos.empty:
    r = top_pos.iloc[0]
    report.append("🟢 TOP POSITIVE:")
    report.append(f"- Rating: {r['reviews.rating']} | Sentiment: {float(r['sentiment_score']):.2f}")
    report.append(f"- {r['reviews.text'][:200]}...\n")

if not top_neg.empty:
    r = top_neg.iloc[0]
    report.append("🔴 TOP NEGATIVE:")
    report.append(f"- Rating: {r['reviews.rating']} | Sentiment: {float(r['sentiment_score']):.2f}")
    report.append(f"- {r['reviews.text'][:200]}...")
report.append("")

# 6. Monthly Sentiment Summary
report.append("6️⃣ MONTHLY SENTIMENT TREND:\n")
if 'reviews.date' in df.columns:
    df['reviews.date'] = pd.to_datetime(df['reviews.date'], errors='coerce')
    monthly_sentiment = df.groupby(df['reviews.date'].dt.to_period('M'))['sentiment_score'].mean()
    for period, score in monthly_sentiment.items():
        report.append(f"- {period}: Avg Sentiment = {score:.2f}")
else:
    report.append("⚠️ Date column missing in dataset.")
report.append("")

# 7. Final Summary
report.append("7️⃣ FINAL INSIGHTS:\n")
avg_score = df['sentiment_score'].mean() if not df.empty else 0
avg_rating = df['reviews.rating'].mean() if 'reviews.rating' in df.columns else 0
report.append(f"- Overall Avg Sentiment Score: {avg_score:.2f}")
report.append(f"- Overall Avg Rating: {avg_rating:.2f}")
report.append("📌 Conclusion: Overall customer satisfaction is moderate to high with improvement areas.")

# Save Report
with open('project_report.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(report))

print("✅ Full project report generated: project_report.txt")
