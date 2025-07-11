import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from fpdf import FPDF
import seaborn as sns
import os
import joblib

# Load full dataset
full_df = pd.read_csv("data/final_reviews.csv", encoding="utf-8")
full_df['reviews.date'] = pd.to_datetime(full_df['reviews.date'])

# Load ML model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# GUI App Setup
root = tk.Tk()
root.title("🛍️ Product Review Sentiment Analyzer")
root.geometry("960x780")

frame_top = tk.Frame(root)
frame_top.pack(pady=10)

# Product Dropdown
tk.Label(frame_top, text="Select Product:").pack(side=tk.LEFT, padx=5)
product_list = full_df['name'].unique().tolist()
selected_product = tk.StringVar()
product_dropdown = ttk.Combobox(frame_top, textvariable=selected_product, values=product_list, width=70)
product_dropdown.pack(side=tk.LEFT, padx=5)
product_dropdown.current(0)

frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=5)

summary_label = tk.Label(root, text="", font=("Helvetica", 12, "bold"), wraplength=900)
summary_label.pack(pady=10)

ml_label = tk.Label(root, text="", font=("Courier", 10), justify=tk.LEFT, wraplength=900)
ml_label.pack(pady=10)

frame_graphs = tk.Frame(root)
frame_graphs.pack(pady=20)

# Show sentiment distribution
def show_sentiment_distribution(df):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(data=df, x='sentiment_label', palette='Set2', ax=ax)
    ax.set_title('Sentiment Distribution')
    canvas = FigureCanvasTkAgg(fig, master=frame_graphs)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Show average monthly sentiment trend
def show_sentiment_trend(df):
    monthly = df.groupby(df['reviews.date'].dt.to_period('M'))['sentiment_score'].mean()
    fig, ax = plt.subplots(figsize=(5, 3))
    monthly.plot(ax=ax, marker='o', color='purple')
    ax.set_title('📈 Monthly Avg Sentiment Trend')
    ax.set_xlabel('Month')
    ax.set_ylabel('Sentiment Score')
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame_graphs)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Show average rating by sentiment
def show_rating_by_sentiment(df):
    avg_rating = df.groupby('sentiment_label')['reviews.rating'].mean()
    fig, ax = plt.subplots(figsize=(4, 3))
    avg_rating.plot(kind='bar', color=['#98FB98', '#FF7F7F', '#87CEFA'], ax=ax)
    ax.set_title('⭐ Avg Rating by Sentiment')
    ax.set_ylabel('Avg Rating')
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame_graphs)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Export text report
def export_txt():
    try:
        lines = [
            "🧠 Sentiment Report",
            "---------------------------------",
            f"Total Reviews: {len(full_df)}",
            f"Positive: {(full_df['sentiment_label']=='Positive').sum()}",
            f"Negative: {(full_df['sentiment_label']=='Negative').sum()}",
            f"Neutral: {(full_df['sentiment_label']=='Neutral').sum()}"
        ]
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            messagebox.showinfo("Success", "TXT Report saved!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Export PDF report
def export_pdf():
    try:
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf")
        if not file_path:
            return

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Product Sentiment Analysis Report", ln=True, align='C')
        pdf.ln(10)

        sentiment_counts = full_df['sentiment_label'].value_counts()
        for label, count in sentiment_counts.items():
            pdf.cell(200, 10, txt=f"{label}: {count} reviews", ln=True)

        avg_rating = full_df['reviews.rating'].mean()
        avg_sentiment = full_df['sentiment_score'].mean()
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Average Rating: {avg_rating:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Average Sentiment Score: {avg_sentiment:.2f}", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Sample Reviews:", ln=True)

        for i, row in full_df.head(5).iterrows():
            pdf.multi_cell(0, 10, txt=f"- ({row['sentiment_label']}) {row['reviews.text']}")

        pdf.output(file_path)
        messagebox.showinfo("Success", "PDF Report Generated!")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Refresh function with summary and ML integration
def refresh_visuals():
    for widget in frame_graphs.winfo_children():
        widget.destroy()

    product = selected_product.get()
    filtered = full_df[full_df['name'] == product]

    if filtered.empty:
        messagebox.showerror("Error", "No reviews found for this product.")
        return

    avg_rating = filtered['reviews.rating'].mean()
    total = len(filtered)

    if avg_rating >= 4:
        tone = "mostly positively"
        emoji = "😊"
    elif avg_rating >= 3:
        tone = "mixed"
        emoji = "😐"
    else:
        tone = "mostly negatively"
        emoji = "☹️"

    summary_text = f"{emoji} This product is {tone} reviewed with an average rating of {avg_rating:.2f} based on {total} reviews."
    summary_label.config(text=summary_text)

    sample_reviews = filtered['reviews.text'].head(5).tolist()
    predictions = model.predict(vectorizer.transform(sample_reviews))
    ml_output = "\n".join([
        f"📝 \"{review[:50]}...\" → 📊 {pred}" for review, pred in zip(sample_reviews, predictions)
    ])
    ml_label.config(text="🤖 ML Model Predictions on Sample Reviews:\n" + ml_output)

    show_sentiment_distribution(filtered)
    show_rating_by_sentiment(filtered)
    show_sentiment_trend(filtered)

# Buttons
btn_refresh = tk.Button(frame_buttons, text="🔍 Analyze Product", command=refresh_visuals)
btn_refresh.pack(side=tk.LEFT, padx=5)

btn_txt = tk.Button(frame_buttons, text="Export TXT Report", command=export_txt)
btn_txt.pack(side=tk.LEFT, padx=5)

btn_pdf = tk.Button(frame_buttons, text="Export PDF Report", command=export_pdf)
btn_pdf.pack(side=tk.LEFT, padx=5)

# Default visualization on startup
refresh_visuals()

root.mainloop()
