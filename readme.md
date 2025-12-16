# Restaurant Review Analytics Dashboard

A **Streamlit-based NLP dashboard** that analyzes restaurant customer reviews to extract **sentiment, aspect-level insights, trends, and actionable business intelligence**. The project is designed to be **academically sound**, **industry-aligned**, and **portfolio-ready**.

---

##  Project Overview

Restaurants receive thousands of customer reviews across platforms. Manually analyzing them is time-consuming and subjective. This project automates the analysis by applying **Natural Language Processing (NLP)** techniques to:

- Classify customer sentiment (Positive / Negative / Neutral)
- Identify key restaurant aspects (food, service, ambience, price, cleanliness)
- Track sentiment trends over time
- Highlight strengths and areas for improvement
- Provide downloadable, processed insights for decision-makers

The dashboard is built using **Python, NLTK, TextBlob, Plotly, and Streamlit**.

---

##  Key Features

###  Sentiment Analysis
- Polarity-based sentiment scoring using **TextBlob**
- Three-class classification: Positive, Negative, Neutral
- Overall sentiment distribution visualization

###  Aspect-Based Analysis
- Rule-based extraction of restaurant aspects:
  - Food
  - Service
  - Ambience
  - Price
  - Cleanliness
- Aspect-wise sentiment scores
- Identification of strongest and weakest aspects

###  Interactive Dashboard
- Clean, manager-friendly UI
- Key metrics at a glance
- Interactive charts using Plotly
- Expandable advanced sections

###  Word Cloud Insights
- Buzz words from:
  - All reviews
  - Positive reviews
  - Negative reviews

###  Trend Analysis
- Monthly sentiment trends
- Visual tracking of customer perception over time

###  Data Export
- Download processed reviews
- Download aspect-level sentiment analysis

---

##  Project Structure

```
restaurant-review-analytics/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── sample_data.csv        # (Optional) Sample input file
```

---

## Expected Input Format

The dashboard accepts a **CSV file** with the following columns:

| Column Name | Description |
|------------|-------------|
| Review     | Customer review text |
| Rating     | Numerical rating (e.g., 1–5) |
| Time       | Review timestamp (date or datetime) |

---

## Tech Stack

- **Python 3.9+**
- **Streamlit** – dashboard & deployment
- **NLTK** – text preprocessing
- **TextBlob** – sentiment analysis
- **Pandas** – data manipulation
- **Plotly** – interactive visualizations
- **Matplotlib & WordCloud** – text visualization

---

## Important Notes

- Sentiment analysis uses **TextBlob**, which:
  - Is fast and interpretable
  - May not fully capture sarcasm or complex context
- This model is intended as a **baseline NLP solution**
- The architecture allows easy upgrades to:
  - Machine Learning models (Logistic Regression, SVM)
  - Deep Learning models (BERT, RoBERTa)

---

## Possible Enhancements

- Replace TextBlob with ML-based sentiment classifier
- Add rating prediction (1–5 stars)
- User authentication & multi-restaurant support
- Deployment on **Streamlit Cloud**
- Integration with Google / Yelp review scraping

---

##  Academic & Portfolio Value

This project demonstrates:
- Applied NLP skills
- Dashboard design & UX thinking
- Data cleaning & preprocessing
- Explainable sentiment analysis
- Business-oriented insights

Ideal for:
- MSc Data Science / AI coursework
- NLP mini-projects
- Portfolio & GitHub showcase

---

##  Author

**Nikhil Kumar**  
MSc Data Science, Artificial Intelligence and Digital Business
Gisma Univeristy of Applied Sciences
Aspiring AI Engineer

---

## License

This project is for **educational and portfolio purposes**. You are free to modify and extend it.

---

*If you found this project useful, consider starring the repository or building on top of it!*

