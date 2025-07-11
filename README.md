# 🛍️ ProductSentimentSuite

**ProductSentimentSuite** is a Python-based sentiment analysis toolkit designed to analyze and visualize product reviews using Machine Learning (ML) and Natural Language Processing (NLP). It includes a GUI built with Tkinter for an interactive experience.

> 🚀 Ideal for business analysts, product managers, and developers looking to explore consumer sentiments in depth.

---

## ✨ Features

- 📊 **Visual Dashboard:** Interactive graphs including sentiment distribution, average ratings, and monthly trends.
- 🧠 **ML Sentiment Classifier:** Trained model classifies reviews into Positive, Negative, or Neutral.
- 🔍 **Product-wise Filtering:** Analyze sentiments for individual products using dropdowns.
- 📝 **Report Generation:** Export results in TXT and PDF format with summary & sample reviews.
- 🎨 **Tabbed GUI Layout:** Tabs for Visualizations, Reports, and Review Explorer.
- 📦 **Modular Structure:** Clean folder structure for easy collaboration and scalability.

---

## 📁 Folder Structure

ProductSentimentSuite/
│
├── data/
│ ├── final_reviews.csv # Processed reviews with sentiment scores
│
├── gui/
│ └── main_gui.py # Tkinter GUI for visualization and interaction
│
├── model/
│ ├── model.pkl # Trained ML model
│ └── vectorizer.pkl # TF-IDF Vectorizer
│
├── scripts/
│ ├── train_model.py # Script to train and evaluate ML model
│ └── sentiment_analysis.py # Sentiment scoring + export visuals
│
└── README.md # This file


---

## 🚀 How to Run

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
pip install pandas matplotlib seaborn textblob sklearn joblib fpdf

🧠 Train the ML Model

python scripts/train_model.py

 Launch the GUI

python gui/main_gui.py

📈 Sample Visuals
Sentiment Distribution

Monthly Sentiment Trends

Average Rating by Sentiment

Product-wise Summary

✅ To-Do / Roadmap
 Add ML-based sentiment analysis

 Build report export options (TXT + PDF)

 Modular GUI with multiple tabs

 Add option to import new review CSVs from GUI

 Deploy as a standalone .exe (via PyInstaller)

🧠 Tech Stack
Frontend: Tkinter

Backend: Python, Pandas, TextBlob, scikit-learn

ML Model: TF-IDF + Logistic Regression

Reports: FPDF

🙌 Acknowledgements
Special thanks to the open-source community for providing awesome libraries and inspiration.

📬 Contact
Developed by Krishanu Mahapatra
📧 GitHub: @krishanu2
