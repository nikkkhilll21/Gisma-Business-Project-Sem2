import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
from datetime import datetime
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Page configuration
st.set_page_config(
    page_title="Restaurant Review Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize NLTK data
@st.cache_resource
def initialize_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass

initialize_nltk()

# Helper functions
def get_sentiment_polarity(text):
    """Calculate sentiment polarity"""
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0

def classify_sentiment(polarity):
    """Classify sentiment"""
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def preprocess_text(text):
    """Clean text for word cloud"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    custom_stops = {'restaurant', 'place', 'one', 'really', 'good', 'great', 'nice'}
    stop_words.update(custom_stops)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def extract_aspects(text):
    """Extract restaurant aspects"""
    aspects = {
        'food': ['food', 'dish', 'meal', 'taste', 'flavor', 'delicious', 'tasty', 'cuisine'],
        'service': ['service', 'staff', 'waiter', 'server', 'manager', 'friendly', 'helpful'],
        'ambience': ['ambience', 'atmosphere', 'decor', 'ambiance', 'seating', 'interior'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'money'],
        'cleanliness': ['clean', 'hygiene', 'hygienic', 'dirty', 'neat', 'tidy']
    }
    
    text_lower = text.lower()
    found_aspects = []
    for aspect, keywords in aspects.items():
        if any(keyword in text_lower for keyword in keywords):
            found_aspects.append(aspect)
    return found_aspects if found_aspects else ['general']

def process_uploaded_file(df):
    """Process uploaded CSV file"""
    try:
        # Check if required columns exist
        required_cols = ['Review', 'Rating', 'Time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info(f"Available columns: {df.columns.tolist()}")
            return None
        
        # Clean and prepare data
        df_clean = df[['Review', 'Rating', 'Time']].copy()
        df_clean['Review'] = df_clean['Review'].fillna('')
        
        # Convert Rating to numeric
        df_clean['Rating'] = pd.to_numeric(df_clean['Rating'], errors='coerce')
        
        # Try multiple date formats with detailed error tracking
        def parse_date_flexible(date_str):
            """Try multiple date formats"""
            if pd.isna(date_str) or date_str == '':
                return pd.NaT
            
            # Convert to string if not already
            date_str = str(date_str).strip()
            
            date_formats = [
                '%m/%d/%Y %H:%M',      # 5/25/2019 15:54
                '%d/%m/%Y %H:%M',      # 25/5/2019 15:54
                '%Y-%m-%d %H:%M:%S',   # 2019-05-25 15:54:00
                '%Y-%m-%d %H:%M',      # 2019-05-25 15:54
                '%m/%d/%Y',            # 5/25/2019
                '%d/%m/%Y',            # 25/5/2019
                '%Y-%m-%d',            # 2019-05-25
                '%m-%d-%Y',            # 5-25-2019
                '%d-%m-%Y',            # 25-5-2019
            ]
            
            for fmt in date_formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue
            
            # If all formats fail, try pandas auto-parsing
            try:
                return pd.to_datetime(date_str, infer_datetime_format=True)
            except:
                return pd.NaT
        
        # Show progress for date parsing
        with st.spinner("Parsing dates..."):
            df_clean['Time'] = df_clean['Time'].apply(parse_date_flexible)
        
        # Count how many dates were successfully parsed
        valid_dates = df_clean['Time'].notna().sum()
        total_dates = len(df_clean)
        
        st.info(f"✓ Successfully parsed {valid_dates}/{total_dates} dates")
        
        if valid_dates == 0:
            st.error("❌ Could not parse any dates. Please check the Time column format.")
            st.write("**Sample Time values:**")
            st.write(df['Time'].head(10).tolist())
            return None
        
        # Remove invalid rows
        df_clean = df_clean[df_clean['Time'].notna()]
        df_clean = df_clean[df_clean['Review'].str.strip() != '']
        df_clean = df_clean[df_clean['Rating'].notna()]
        
        if len(df_clean) == 0:
            st.error("No valid data after cleaning.")
            return None
        
    except Exception as e:
        st.error(f"Error in process_uploaded_file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None
    
    # Add date features
    df_clean['Year_Month'] = df_clean['Time'].dt.to_period('M').astype(str)
    df_clean['Month_Name'] = df_clean['Time'].dt.strftime('%B %Y')
    
    # Sentiment analysis
    df_clean['Sentiment_Score'] = df_clean['Review'].apply(get_sentiment_polarity)
    df_clean['Sentiment'] = df_clean['Sentiment_Score'].apply(classify_sentiment)
    
    # Text preprocessing
    df_clean['Processed_Review'] = df_clean['Review'].apply(preprocess_text)
    
    # Aspect extraction
    df_clean['Aspects'] = df_clean['Review'].apply(extract_aspects)
    
    return df_clean

def create_aspect_dataframe(df_clean):
    """Create detailed aspect analysis"""
    aspect_data = []
    aspects_dict = {
        'food': ['food', 'dish', 'meal', 'taste', 'flavor', 'delicious', 'tasty', 'cuisine'],
        'service': ['service', 'staff', 'waiter', 'server', 'manager', 'friendly', 'helpful'],
        'ambience': ['ambience', 'atmosphere', 'decor', 'ambiance', 'seating', 'interior'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'money'],
        'cleanliness': ['clean', 'hygiene', 'hygienic', 'dirty', 'neat', 'tidy']
    }
    
    for idx, row in df_clean.iterrows():
        for aspect in row['Aspects']:
            if aspect != 'general':
                # Get aspect-specific sentiment
                text_lower = row['Review'].lower()
                sentences = row['Review'].split('.')
                aspect_sentences = [s for s in sentences if any(kw in s.lower() for kw in aspects_dict[aspect])]
                
                if aspect_sentences:
                    combined = ' '.join(aspect_sentences)
                    sentiment_score = TextBlob(combined).sentiment.polarity
                else:
                    sentiment_score = row['Sentiment_Score']
                
                aspect_data.append({
                    'Month': row['Month_Name'],
                    'Year_Month': row['Year_Month'],
                    'Aspect': aspect,
                    'Sentiment_Score': sentiment_score,
                    'Sentiment': classify_sentiment(sentiment_score),
                    'Review': row['Review']
                })
    
    if len(aspect_data) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(aspect_data)

# Main app
def main():
    st.title("🍽️ Restaurant Review Analytics Dashboard")
    st.markdown("Upload your restaurant reviews CSV file to get comprehensive insights!")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file with Reviews, Ratings, and Time",
            type=['csv'],
            help="CSV must contain: Review, Rating, Time columns"
        )
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info(
            "This dashboard analyzes restaurant reviews to provide:\n"
            "- Sentiment distribution\n"
            "- Aspect analysis\n"
            "- Word clouds\n"
            "- Monthly trends\n"
            "- Actionable insights"
        )
    
    # Main content
    if uploaded_file is not None:
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Load data with error handling
            st.write("### 📂 Loading Data...")
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                    st.info("ℹ️ File loaded with Latin-1 encoding")
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                    st.info("ℹ️ File loaded with ISO-8859-1 encoding")
            
            st.success(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Show data preview - ALWAYS EXPANDED so user can see the data
            with st.expander("👀 View Uploaded Data", expanded=True):
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {df.columns.tolist()}")
                
                # Check for column name issues
                if any(' ' in str(col) for col in df.columns):
                    st.warning("⚠️ Column names have extra spaces. Auto-cleaning...")
                    df.columns = df.columns.str.strip()
                
                st.dataframe(df.head(10))
                
                st.write("**Data Types:**")
                st.write(df.dtypes)
                
                if 'Time' in df.columns:
                    st.write("**Sample Time values (first 10 rows):**")
                    time_samples = df['Time'].head(10).tolist()
                    for i, val in enumerate(time_samples, 1):
                        st.text(f"{i}. {val}")
                else:
                    st.error("❌ 'Time' column not found!")
                    st.write("Please rename your date/time column to 'Time'")
                    st.stop()
            
            # Process data
            st.write("### ⚙️ Processing Data...")
            df_clean = process_uploaded_file(df)
            
            if df_clean is None:
                st.error("❌ Processing failed. Please check the messages above.")
                st.stop()
            
            st.write("### 🔍 Analyzing Aspects...")
            df_aspects = create_aspect_dataframe(df_clean)
            
            st.success(f"✅ Successfully processed {len(df_clean)} reviews!")
            
            # Key Metrics
            st.header("📊 Key Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Reviews", len(df_clean))
            with col2:
                st.metric("Avg Rating", f"{df_clean['Rating'].mean():.2f}⭐")
            with col3:
                positive_pct = (len(df_clean[df_clean['Sentiment'] == 'Positive']) / len(df_clean) * 100)
                st.metric("Positive", f"{positive_pct:.1f}%")
            with col4:
                negative_pct = (len(df_clean[df_clean['Sentiment'] == 'Negative']) / len(df_clean) * 100)
                st.metric("Negative", f"{negative_pct:.1f}%")
            with col5:
                st.metric("Avg Sentiment", f"{df_clean['Sentiment_Score'].mean():.2f}")
            
            # Sentiment Distribution
            st.header("😊 Positive vs Negative Distribution")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                sentiment_counts = df_clean['Sentiment'].value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Rating Distribution")
                rating_counts = df_clean['Rating'].value_counts().sort_index()
                fig = px.bar(
                    x=rating_counts.index,
                    y=rating_counts.values,
                    labels={'x': 'Rating', 'y': 'Count'},
                    color=rating_counts.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Aspect Analysis
            st.header("🔍 What's Right & What's Wrong?")
            
            if len(df_aspects) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Most Discussed Aspects")
                    aspect_counts = df_aspects['Aspect'].value_counts()
                    fig = px.bar(
                        x=aspect_counts.values,
                        y=aspect_counts.index,
                        orientation='h',
                        labels={'x': 'Number of Mentions', 'y': 'Aspect'},
                        color=aspect_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Aspect Sentiment Scores")
                    aspect_sentiment = df_aspects.groupby('Aspect')['Sentiment_Score'].mean().sort_values()
                    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in aspect_sentiment.values]
                    
                    fig = go.Figure(go.Bar(
                        x=aspect_sentiment.values,
                        y=aspect_sentiment.index,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{x:.2f}" for x in aspect_sentiment.values],
                        textposition='auto'
                    ))
                    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
                    fig.update_layout(
                        xaxis_title="Average Sentiment Score",
                        yaxis_title="Aspect",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed aspect insights
                st.subheader("📋 Detailed Aspect Analysis")
                
                # What's right
                positive_aspects = df_aspects[df_aspects['Sentiment'] == 'Positive'].groupby('Aspect').size().sort_values(ascending=False)
                if len(positive_aspects) > 0:
                    st.success(f"**✅ Strengths:** {positive_aspects.index[0].capitalize()} is highly praised!")
                
                # What's wrong
                negative_aspects = df_aspects[df_aspects['Sentiment'] == 'Negative'].groupby('Aspect').size().sort_values(ascending=False)
                if len(negative_aspects) > 0:
                    st.error(f"**⚠️ Needs Improvement:** {negative_aspects.index[0].capitalize()} has received negative feedback")
                
                # Show sample reviews for each sentiment
                aspect_filter = st.selectbox("Select aspect to view sample reviews:", 
                                            ['All'] + sorted(df_aspects['Aspect'].unique().tolist()))
                
                if aspect_filter != 'All':
                    filtered_reviews = df_aspects[df_aspects['Aspect'] == aspect_filter]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🟢 Positive Reviews:**")
                        pos_reviews = filtered_reviews[filtered_reviews['Sentiment'] == 'Positive']['Review'].head(3)
                        for i, review in enumerate(pos_reviews, 1):
                            st.text_area(f"Review {i}", review[:200] + "...", height=100, key=f"pos_{i}")
                    
                    with col2:
                        st.markdown("**🔴 Negative Reviews:**")
                        neg_reviews = filtered_reviews[filtered_reviews['Sentiment'] == 'Negative']['Review'].head(3)
                        for i, review in enumerate(neg_reviews, 1):
                            st.text_area(f"Review {i}", review[:200] + "...", height=100, key=f"neg_{i}")
            
            # Word Clouds
            st.header("☁️ Buzz Words - What People Are Saying")
            
            col1, col2, col3 = st.columns(3)
            
            # All reviews
            with col1:
                st.subheader("All Reviews")
                all_text = ' '.join(df_clean['Processed_Review'])
                if len(all_text) > 0:
                    wordcloud = WordCloud(width=400, height=400, 
                                        background_color='white',
                                        colormap='viridis').generate(all_text)
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            
            # Positive reviews
            with col2:
                st.subheader("Positive Reviews")
                positive_text = ' '.join(df_clean[df_clean['Sentiment'] == 'Positive']['Processed_Review'])
                if len(positive_text) > 0:
                    wordcloud = WordCloud(width=400, height=400,
                                        background_color='white',
                                        colormap='Greens').generate(positive_text)
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            
            # Negative reviews
            with col3:
                st.subheader("Negative Reviews")
                negative_text = ' '.join(df_clean[df_clean['Sentiment'] == 'Negative']['Processed_Review'])
                if len(negative_text) > 0:
                    wordcloud = WordCloud(width=400, height=400,
                                        background_color='white',
                                        colormap='Reds').generate(negative_text)
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            
            # Top keywords
            st.subheader("🔑 Top Keywords")
            all_words = ' '.join(df_clean['Processed_Review']).split()
            word_freq = Counter(all_words).most_common(20)
            
            fig = px.bar(
                x=[w[1] for w in word_freq],
                y=[w[0] for w in word_freq],
                orientation='h',
                labels={'x': 'Frequency', 'y': 'Word'},
                title="Top 20 Most Frequent Words"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly Trends
            st.header("📈 Monthly Sentiment Trends")
            
            if len(df_aspects) > 0:
                # Overall sentiment trend
                monthly_sentiment = df_clean.groupby('Month_Name')['Sentiment_Score'].mean().reset_index()
                monthly_sentiment = monthly_sentiment.sort_values('Month_Name')
                
                fig = px.line(
                    monthly_sentiment,
                    x='Month_Name',
                    y='Sentiment_Score',
                    title="Overall Sentiment Trend Over Time",
                    markers=True
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.update_layout(xaxis_title="Month", yaxis_title="Average Sentiment Score")
                st.plotly_chart(fig, use_container_width=True)
                
                # Aspect-wise monthly trend
                st.subheader("Aspect-wise Monthly Trends")
                aspect_monthly = df_aspects.groupby(['Month', 'Aspect'])['Sentiment_Score'].mean().reset_index()
                
                fig = px.line(
                    aspect_monthly,
                    x='Month',
                    y='Sentiment_Score',
                    color='Aspect',
                    title="Sentiment Trends by Aspect",
                    markers=True
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)
            
            # Overall Summary
            st.header("📝 Executive Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Key Strengths")
                if len(df_aspects) > 0:
                    best_aspect = df_aspects.groupby('Aspect')['Sentiment_Score'].mean().idxmax()
                    best_score = df_aspects.groupby('Aspect')['Sentiment_Score'].mean().max()
                    st.success(f"**{best_aspect.capitalize()}** is the strongest aspect (Score: {best_score:.2f})")
                
                positive_pct = len(df_clean[df_clean['Sentiment'] == 'Positive']) / len(df_clean) * 100
                if positive_pct > 70:
                    st.success(f"**{positive_pct:.1f}%** of reviews are positive - Great job!")
            
            with col2:
                st.subheader("⚠️ Areas for Improvement")
                if len(df_aspects) > 0:
                    worst_aspect = df_aspects.groupby('Aspect')['Sentiment_Score'].mean().idxmin()
                    worst_score = df_aspects.groupby('Aspect')['Sentiment_Score'].mean().min()
                    st.warning(f"**{worst_aspect.capitalize()}** needs attention (Score: {worst_score:.2f})")
                
                negative_pct = len(df_clean[df_clean['Sentiment'] == 'Negative']) / len(df_clean) * 100
                if negative_pct > 20:
                    st.warning(f"**{negative_pct:.1f}%** of reviews are negative - Focus area!")
            
            # Download processed data
            st.header("💾 Download Processed Data")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Processed Reviews",
                    data=csv,
                    file_name="processed_reviews.csv",
                    mime="text/csv"
                )
            
            with col2:
                if len(df_aspects) > 0:
                    csv_aspects = df_aspects.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Aspect Analysis",
                        data=csv_aspects,
                        file_name="aspect_analysis.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            
            # Show detailed debugging
            with st.expander("🔍 Full Error Details", expanded=True):
                import traceback
                st.code(traceback.format_exc())
                
                st.write("---")
                st.write("**Troubleshooting Guide:**")
                st.markdown("""
                **Common issues and solutions:**
                
                1. **"Could not convert string" error:**
                   - Check if your Time column has the correct date format
                   - Expected formats: `5/25/2019 15:54` or `2019-05-25 15:54`
                   - Look at the "Sample Time values" above
                
                2. **Column not found:**
                   - Make sure you have columns named exactly: `Review`, `Rating`, `Time`
                   - Column names are case-sensitive
                   - Check for extra spaces in column names
                
                3. **Encoding issues:**
                   - Try saving your CSV with UTF-8 encoding
                   - Open in Excel → Save As → CSV UTF-8
                
                4. **Still not working?**
                   - Download the sample CSV below and try that first
                   - Then format your data to match the sample
                """)
                
                # Create downloadable sample
                sample_df = pd.DataFrame({
                    'Review': [
                        'Great food and excellent ambience! The staff was very friendly.',
                        'Service was slow but food was absolutely delicious. Would recommend.',
                        'Terrible experience. Food was cold and staff was rude.',
                        'Amazing restaurant! Good value for money and clean environment.',
                        'Average food, nothing special. Ambience was nice though.'
                    ],
                    'Rating': [5, 4, 1, 5, 3],
                    'Time': ['5/25/2019 15:54', '5/24/2019 22:11', '5/23/2019 18:30', '5/22/2019 20:15', '5/21/2019 19:00']
                })
                
                csv_sample = sample_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Working Sample CSV",
                    data=csv_sample,
                    file_name="sample_restaurant_reviews.csv",
                    mime="text/csv",
                    help="Download this sample file to see the correct format"
                )
    
    else:
        st.info("👆 Please upload a CSV file to begin analysis")
        
        # Show sample data format
        st.subheader("📋 Expected CSV Format")
        sample_data = pd.DataFrame({
            'Review': ['Great food and ambience!', 'Service was slow but food was good'],
            'Rating': [5, 3],
            'Time': ['5/25/2019 15:54', '5/24/2019 22:11']
        })
        st.dataframe(sample_data)

if __name__ == "__main__":
    main()