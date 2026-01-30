
import pandas as pd
import yfinance as yf
import os
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

ticker = "CAT"

def collect_news_data(ticker='CAT', max_items=5000):
    """
    Reads news data from a local RSS feed file (.rss.xml).
    """
    print(f"Collecting news data from local file '{ticker}.rss.xml'...")

    rss_file_name = f"{ticker}.rss.xml" # Assuming the user uploaded this file

    try:
        # Check if the file exists
        if not os.path.exists(rss_file_name):
            print(f"Error: RSS file '{rss_file_name}' not found in the current directory. Please upload it.")
            return pd.DataFrame(columns=['Source', 'Headline', 'PubDate'])

        # Read the content of the local RSS file
        with open(rss_file_name, 'r', encoding='utf-8') as f:
            rss_content = f.read()

        soup = BeautifulSoup(rss_content, "xml") # Parse the content
        items = soup.find_all("item")

        data = []
        for i, item in enumerate(items[:max_items], 1):
            try:
                pub_date = item.pubDate.text if item.pubDate else None
                headline = item.title.text if item.title else None
                source = item.link.text if item.link else None

                data.append({
                    'Source': source,
                    'Headline': headline,
                    'PubDate': pub_date
                })
            except Exception as e:
                print(f"Error parsing item {i}: {e}")
                continue

        df = pd.DataFrame(data)
        print(f"Successfully collected {len(df)} news items from '{rss_file_name}'")
        return df

    except Exception as e:
        print(f"Error collecting news data from local file '{rss_file_name}': {e}")
        return pd.DataFrame(columns=['Source', 'Headline', 'PubDate'])


def clean_news_data(df):
    """
    Clean and process news data
    """
    print("Cleaning news data...")

    if df.empty:
        print("Warning: Empty dataframe provided")
        return df

    # Parse publication dates
    df['PubDate_Clean'] = df['PubDate'].str.split(' ').str[1:4].str.join(" ")
    df['PubDate'] = pd.to_datetime(df['PubDate_Clean'], format="%d %b %Y", errors='coerce')
    df['Date'] = df['PubDate'].dt.date

    # Calculate headline length
    df['Headline Length'] = df['Headline'].str.len()

    # Drop temporary column
    df.drop('PubDate_Clean', axis=1, inplace=True, errors='ignore')

    print(f"Cleaned data: {len(df)} rows with dates from {df['Date'].min()} to {df['Date'].max()}")
    return df


def collect_stock_data(ticker='CAT', period='5y'):
    """
    Collect stock price data using yfinance
    """
    print(f"Collecting stock data for {ticker}...")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            print("Warning: No stock data retrieved")
            return pd.DataFrame()

        # Select relevant columns
        df_stock = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Add date column
        df_stock.index = pd.to_datetime(df_stock.index)
        df_stock['Date'] = df_stock.index.date
        df_stock.index = df_stock.index.date

        print(f"Successfully collected {len(df_stock)} days of stock data")
        return df_stock

    except Exception as e:
        print(f"Error collecting stock data: {e}")
        return pd.DataFrame()


def merge_datasets(stock_df, news_df):
    """
    Merge stock and news data on date
    """
    print("Merging datasets...")

    if stock_df.empty or news_df.empty:
        print("Warning: One or both dataframes are empty")
        if not stock_df.empty:
            return stock_df
        return pd.DataFrame()

    # Perform left merge to keep all trading days
    merged_df = stock_df.merge(news_df, on='Date', how='left')

    print(f"Merged dataset: {len(merged_df)} rows")
    print(f"News coverage: {merged_df['Headline'].notna().sum()} days have news")

    return merged_df


def save_datasets(news_raw, news_clean, stock_data, merged_data):
    """
    Save all datasets to CSV files
    """
    print("\nSaving datasets...")

    try:
        if not news_raw.empty:
            news_raw.to_csv('News_raw.csv', index=False)
            print("✓ Saved News_raw.csv")

        if not news_clean.empty:
            news_clean.to_csv('news_cleaned.csv', index=False)
            print("✓ Saved news_cleaned.csv")

        if not stock_data.empty:
            stock_data.to_csv('stock_data.csv')
            print("✓ Saved stock_data.csv")

        if not merged_data.empty:
            merged_data.to_csv('merged_medsem_data.csv', index=False)
            print("✓ Saved merged_medsem_data.csv")

    except Exception as e:
        print(f"Error saving files: {e}")


def generate_summary_statistics(merged_df):
    """
    Generate summary statistics for the report
    """
    print("\n" + "="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)

    if merged_df.empty:
        print("No data to summarize")
        return

    print(f"\nDate Range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    print(f"Total Trading Days: {len(merged_df)}")
    print(f"Days with News: {merged_df['Headline'].notna().sum()}")
    print(f"Days without News: {merged_df['Headline'].isna().sum()}")

    print(f"\nStock Price Statistics:")
    print(f"  Opening Price Range: ${merged_df['Open'].min():.2f} - ${merged_df['Open'].max():.2f}")
    print(f"  Closing Price Range: ${merged_df['Close'].min():.2f} - ${merged_df['Close'].max():.2f}")
    print(f"  Average Daily Volume: {merged_df['Volume'].mean():,.0f}")

    if 'Headline Length' in merged_df.columns:
        print(f"\nNews Statistics:")
        print(f"  Average Headline Length: {merged_df['Headline Length'].mean():.0f} characters")
        print(f"  Shortest Headline: {merged_df['Headline Length'].min():.0f} characters")
        print(f"  Longest Headline: {merged_df['Headline Length'].max():.0f} characters")

    print("\n" + "="*60)


def main():
    """
    Main execution function
    """
    print("="*60)
    print("WiDS 2025 - Stock Market & News Sentiment Analysis")
    print("Data Collection Script")
    print("="*60 + "\n")

    # Configuration
    TICKER = 'CAT'
    PERIOD = '5y'  # Can change to '3mo', '6mo', etc.

    # Step 1: Collect news data
    news_raw = collect_news_data(ticker=TICKER, max_items=5000)

    # Step 2: Clean news data
    news_clean = clean_news_data(news_raw.copy()) if not news_raw.empty else pd.DataFrame()

    # Step 3: Collect stock data
    stock_data = collect_stock_data(ticker=TICKER, period=PERIOD) # Corrected variable name

    # Step 4: Merge datasets
    merged_data = merge_datasets(stock_data, news_clean)

    # Step 5: Save all datasets
    save_datasets(news_raw, news_clean, stock_data, merged_data)

    # Step 6: Generate summary
    generate_summary_statistics(merged_data)

    print("\n✓ Data collection complete!")
    print("\nGenerated files:")
    print("  - News_raw.csv")
    print("  - news_cleaned.csv")
    print("  - stock_data.csv")
    print("  - merged_medsem_data.csv")


if __name__ == "__main__":
    main()

