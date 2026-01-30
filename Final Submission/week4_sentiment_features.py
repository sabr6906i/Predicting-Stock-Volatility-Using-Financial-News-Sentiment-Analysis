
"""
Week 4: Sentiment Analysis & Feature Engineering
WiDS 2025 Project - Stock Volatility Prediction

This script:
1. Performs sentiment analysis on news headlines
2. Engineers features from sentiment scores
3. Prepares data for machine learning models
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """Analyze sentiment of financial news headlines"""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_scores(self, text):
        """
        Get VADER sentiment scores for text
        Returns: compound, positive, negative, neutral scores
        """
        if pd.isna(text):
            return 0.0, 0.0, 0.0, 0.0

        scores = self.analyzer.polarity_scores(str(text))
        return (
            scores['compound'],
            scores['pos'],
            scores['neg'],
            scores['neu']
        )

    def classify_sentiment(self, compound_score):
        """
        Classify sentiment as Positive, Negative, or Neutral
        Based on compound score thresholds
        """
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'


class FeatureEngineer:
    """Engineer features for stock prediction"""

    @staticmethod
    def calculate_price_change(df):
        """Calculate daily price change and direction"""
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Price_Change'] / df['Open']) * 100
        df['Price_Direction'] = df['Price_Change'].apply(
            lambda x: 1 if x > 0 else 0
        )
        return df

    @staticmethod
    def calculate_volatility(df, window=5):
        """Calculate rolling volatility"""
        df['Volatility'] = df['Close'].pct_change().rolling(window=window).std()
        return df

    @staticmethod
    def calculate_momentum(df):
        """Calculate price momentum indicators"""
        df['Momentum_1d'] = df['Close'].pct_change(1)
        df['Momentum_3d'] = df['Close'].pct_change(3)
        df['Momentum_5d'] = df['Close'].pct_change(5)
        return df

    @staticmethod
    def calculate_ma(df, windows=[5, 10]):
        """Calculate moving averages"""
        for window in windows:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        return df

    @staticmethod
    def aggregate_daily_sentiment(df):
        """
        Aggregate sentiment scores for days with multiple news
        """
        sentiment_agg = df.groupby('Date').agg({
            'Sentiment_Compound': ['mean', 'std', 'count'],
            'Sentiment_Positive': 'mean',
            'Sentiment_Negative': 'mean',
            'Sentiment_Neutral': 'mean'
        })

        # Flatten column names
        sentiment_agg.columns = [
            'Sentiment_Mean', 'Sentiment_Std', 'News_Count',
            'Positive_Mean', 'Negative_Mean', 'Neutral_Mean'
        ]

        # Fill std with 0 for days with single news
        sentiment_agg['Sentiment_Std'] = sentiment_agg['Sentiment_Std'].fillna(0)

        return sentiment_agg


def add_sentiment_features(news_df):
    """
    Add sentiment analysis features to news dataframe

    Args:
        news_df: DataFrame with 'Headline' column

    Returns:
        DataFrame with sentiment features added
    """
    print("Performing sentiment analysis...")

    analyzer = SentimentAnalyzer()

    # Apply sentiment analysis to each headline
    sentiment_results = news_df['Headline'].apply(
        lambda x: analyzer.get_sentiment_scores(x)
    )

    # Unpack results into separate columns
    news_df['Sentiment_Compound'] = sentiment_results.apply(lambda x: x[0])
    news_df['Sentiment_Positive'] = sentiment_results.apply(lambda x: x[1])
    news_df['Sentiment_Negative'] = sentiment_results.apply(lambda x: x[2])
    news_df['Sentiment_Neutral'] = sentiment_results.apply(lambda x: x[3])

    # Add sentiment classification
    news_df['Sentiment_Label'] = news_df['Sentiment_Compound'].apply(
        analyzer.classify_sentiment
    )

    print(f"✓ Sentiment analysis complete for {len(news_df)} headlines")

    # Print sentiment distribution
    print("\nSentiment Distribution:")
    print(news_df['Sentiment_Label'].value_counts())
    print(f"\nAverage Sentiment Score: {news_df['Sentiment_Compound'].mean():.3f}")

    return news_df


def engineer_stock_features(stock_df):
    """
    Add technical indicators and features to stock data

    Args:
        stock_df: DataFrame with OHLCV data

    Returns:
        DataFrame with engineered features
    """
    print("\nEngineering stock features...")

    engineer = FeatureEngineer()

    # Price-based features
    stock_df = engineer.calculate_price_change(stock_df)
    stock_df = engineer.calculate_volatility(stock_df)
    stock_df = engineer.calculate_momentum(stock_df)
    stock_df = engineer.calculate_ma(stock_df, windows=[5, 10])

    # Volume features
    stock_df['Volume_MA5'] = stock_df['Volume'].rolling(window=5).mean()
    stock_df['Volume_Change'] = stock_df['Volume'].pct_change()

    print(f"✓ Engineered {len([c for c in stock_df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']])} new features")

    return stock_df


def create_ml_dataset(stock_df, news_df):
    """
    Merge stock and sentiment data, create features for ML

    Args:
        stock_df: DataFrame with stock features
        news_df: DataFrame with sentiment features

    Returns:
        DataFrame ready for machine learning
    """
    print("\nCreating ML-ready dataset...")

    # Ensure Date columns are datetime
    if 'Date' not in stock_df.columns and stock_df.index.name == 'Date':
        stock_df = stock_df.reset_index()
    if 'Date' in news_df.columns:
        news_df['Date'] = pd.to_datetime(news_df['Date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    # Aggregate sentiment by date
    engineer = FeatureEngineer()
    sentiment_agg = engineer.aggregate_daily_sentiment(news_df)

    # Merge with stock data
    ml_df = stock_df.merge(sentiment_agg, left_on='Date', right_index=True, how='left')

    # Fill missing sentiment values (days without news)
    sentiment_cols = ['Sentiment_Mean', 'Sentiment_Std', 'News_Count',
                     'Positive_Mean', 'Negative_Mean', 'Neutral_Mean']
    ml_df[sentiment_cols] = ml_df[sentiment_cols].fillna(0)

    # Create lagged features (previous day sentiment)
    ml_df['Sentiment_Lag1'] = ml_df['Sentiment_Mean'].shift(1)
    ml_df['News_Count_Lag1'] = ml_df['News_Count'].shift(1)

    # Create target variable (next day price direction)
    ml_df['Target_Next_Day'] = ml_df['Price_Direction'].shift(-1)

    # Drop rows with NaN in target or critical features
    ml_df = ml_df.dropna(subset=['Target_Next_Day', 'Volatility', 'MA_5'])

    print(f"✓ Created dataset with {len(ml_df)} samples and {len(ml_df.columns)} features")
    print(f"✓ Target distribution: {ml_df['Target_Next_Day'].value_counts().to_dict()}")

    return ml_df


def save_processed_data(news_df, stock_df, ml_df):
    """Save all processed datasets"""
    print("\nSaving processed datasets...")

    news_df.to_csv('news_with_sentiment.csv', index=False)
    print("✓ Saved news_with_sentiment.csv")

    stock_df.to_csv('stock_with_features.csv', index=False)
    print("✓ Saved stock_with_features.csv")

    ml_df.to_csv('ml_ready_dataset.csv', index=False)
    print("✓ Saved ml_ready_dataset.csv")


def generate_feature_summary(ml_df):
    """Generate summary statistics of features"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)

    print(f"\nDataset Shape: {ml_df.shape}")
    print(f"Date Range: {ml_df['Date'].min()} to {ml_df['Date'].max()}")

    print("\nPrice Statistics:")
    print(f"  Average Daily Return: {ml_df['Price_Change_Pct'].mean():.2f}%")
    print(f"  Volatility (std): {ml_df['Volatility'].mean():.4f}")
    print(f"  Up Days: {(ml_df['Price_Direction'] == 1).sum()}")
    print(f"  Down Days: {(ml_df['Price_Direction'] == 0).sum()}")

    print("\nSentiment Statistics:")
    print(f"  Average Sentiment: {ml_df['Sentiment_Mean'].mean():.3f}")
    print(f"  Days with News: {(ml_df['News_Count'] > 0).sum()}")
    print(f"  Average News per Day: {ml_df['News_Count'].mean():.1f}")

    print("\nFeature Correlation with Target:")
    corr_cols = ['Sentiment_Mean', 'Price_Change_Pct', 'Volatility',
                 'Momentum_1d', 'Volume_Change']
    correlations = ml_df[corr_cols + ['Target_Next_Day']].corr()['Target_Next_Day'].drop('Target_Next_Day')
    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.3f}")

    print("\n" + "="*60)


def main():
    """Main execution function"""
    print("="*60)
    print("Week 4: Sentiment Analysis & Feature Engineering")
    print("="*60 + "\n")

    # Load data
    try:
        news_df = pd.read_csv('news_cleaned.csv')
        stock_df = pd.read_csv('stock_data.csv')
        print(f"✓ Loaded {len(news_df)} news items")
        print(f"✓ Loaded {len(stock_df)} trading days")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the data collection script first!")
        return

    # Step 1: Add sentiment features
    news_df = add_sentiment_features(news_df)

    # Step 2: Engineer stock features
    stock_df = engineer_stock_features(stock_df)

    # Step 3: Create ML dataset
    ml_df = create_ml_dataset(stock_df, news_df)

    # Step 4: Generate summary
    generate_feature_summary(ml_df)

    # Step 5: Save all processed data
    save_processed_data(news_df, stock_df, ml_df)

    print("\n✓ Week 4 Feature Engineering Complete!")
    print("\nNext step: Build prediction models using 'ml_ready_dataset.csv'")


if __name__ == "__main__":
    # Install vader if not present
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("Installing vaderSentiment...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'vaderSentiment'])
        print("✓ Installed vaderSentiment")

    main()
