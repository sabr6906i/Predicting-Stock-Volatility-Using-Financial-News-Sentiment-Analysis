# Midterm Submission

This midterm submission evaluates the understanding of **data collection, parsing, cleaning, and integration**
of financial news and stock market data.

## Objective
- Collect real-world financial news from RSS/XML feeds
- Parse and structure unstructured XML data
- Collect historical stock price data using an API
- Align news data with market trading days
- Reason about missing values and non-trading days
- Perform basic exploratory analysis

---

## Tasks to Complete

### Task 1: Multi-source Financial News Collection
- Collect financial news from different RSS feeds
  (Example sources: Yahoo Finance, Reuters, MarketWatch)
- Extract and store the following fields:
  - `source`
  - `headline`
  - `pubDate`

**Output file:**
```
news_raw.csv
```

---

### Task 2: XML Structure Understanding
Include a **Markdown section** in the notebook answering:
- Which XML tags were used to extract the headlines?
- What is the role of the `<item>` tag in RSS feeds?
- How does an RSS feed differ from a normal HTML webpage?

---

### Task 3: News Data Cleaning and Standardization
- Convert publication timestamps to UTC timezone
- Convert timestamps to date-only format (`YYYY-MM-DD`)
- Add the following columns:
  - `date`
  - `headline_length` (number of characters in the headline)

**Output file:**
```
news_cleaned.csv
```

---

### Task 4: Stock Price Data Collection
- Choose **one US-listed stock**
- Download **at least 10 trading days** of data using `yfinance`
- Required columns:
  - Open
  - High
  - Low
  - Close
  - Volume

**Output file:**
```
stock_data.csv
```

---

### Task 5: Market Calendar Awareness
Include a **Markdown section** answering:
- Which dates in your news data are non-trading days?
- Why does the stock market not trade on those days?
- How many news articles fall on non-trading days?

---

### Task 6: Intelligent Data Merging
- Merge news data with stock price data on `date`
- Keep **all news rows**
- Do **not** drop missing stock values
- Add a boolean column:
  - `is_trading_day` (True / False)

**Output file:**
```
merged_midterm_data.csv
```

## Submission Requirements
Submit a link to github repo with the following:
- Jupyter Notebook:
```
midterm_submission.ipynb
```
- CSV files:
```
news_raw.csv
news_cleaned.csv
merged_midterm_data.csv
```
