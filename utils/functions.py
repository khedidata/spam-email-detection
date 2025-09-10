import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud


url_pattern = re.compile(r"https?://\S+|www\.\S+")
email_pattern = re.compile(r"\S+@\S+")
punc_pattern = re.compile(r"[^a-z\s]")

stop_words = set(stopwords.words('english'))

def loading_dataset(data_folder_name: str = "data/", csv_file: str = "email.csv") -> pd.DataFrame:
    """
    Load and preprocess the email dataset from a CSV file.

    This function performs the following steps:
    1. Checks if there is at least one CSV file in the specified data folder.
    2. Loads the CSV file into a pandas DataFrame.
    3. Converts all column names to lowercase for consistency.
    4. Filters the dataset to include only the 'ham' and 'spam' categories.

    Parameters
    ----------
    data_folder_name : str, optional
        Path to the folder containing the dataset CSV file. Default is 'data/'.
    csv_file : str, optional
        Name of the CSV file to load. Default is 'email.csv'.

    Returns
    -------
    pd.DataFrame
        A preprocessed DataFrame containing only emails labeled as 'ham' or 'spam'.

    Raises
    ------
    AssertionError
        If there is no CSV file in the specified folder.
    """
    # Check that at least one CSV file exists in the folder
    assert any(f.endswith(".csv") for f in os.listdir(data_folder_name)), \
        "Initial working file not found in the data folder!"
    
    # Load the CSV
    data = pd.read_csv(os.path.join(data_folder_name, csv_file))
    
    # Standardize column names
    data.columns = [c.lower() for c in data.columns]
    # Keep only relevant categories
    categories_list = ['ham', 'spam']
    data = data[data["category"].isin(categories_list)]
    
    return data

def generate_email_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of email statistics for a binary classification dataset.

    This function computes descriptive statistics for emails labeled as 'spam' or 'ham',
    including counts, percentages, and message length statistics (mean, median, std).
    It returns a formatted summary table suitable for quick reporting.

    Parameters
    ----------
    data : pd.DataFrame
        The email dataset containing at least the columns:
        - "category" : labels for emails ('spam' or 'ham')
        - "message" : the text content of the emails

    Returns
    -------
    pd.DataFrame
        A summary table with one row per statistic and a single column "Informations",
        including:
        - Total number of emails
        - Email categories
        - Percentage and count of 'spam' and 'ham'
        - Average, median, and standard deviation of message length for 'spam' and 'ham'
    
    Notes
    -----
    - Assumes that the "category" column contains only 'spam' or 'ham' values.
    - The message lengths are calculated in number of characters.
    """
    # Copy the dataset to avoid modifying original
    email = data.copy()

    # Compute message lengths
    email["message_len"] = email["message"].str.len()

    # Boolean columns for categories
    email["is_spam"] = email["category"].str.lower().eq("spam")
    email["is_ham"] = email["category"].str.lower().eq("ham")
    email["is_nan_message"] = email["category"].isna()

    # Group by category and compute basic statistics
    stats = email.groupby("category").agg(
        n_msg=("message", "size"),
        n_nan=("is_nan_message", "sum"),
        mean_len_msg=("message_len", "mean"),
        median_len_msg=("message_len", "median"),
        std_len_msg=("message_len", "std")
    )

    # Counts and percentages
    n_spam = email["is_spam"].sum()
    n_ham = email["is_ham"].sum()
    pct_spam = n_spam / len(email)
    pct_ham = n_ham / len(email)

    # Message length statistics
    spam_stat = stats.loc["spam"]
    mean_len_spam = spam_stat["mean_len_msg"]
    median_len_spam = spam_stat["median_len_msg"]
    std_len_spam = spam_stat["std_len_msg"]

    ham_stat = stats.loc["ham"]
    mean_len_ham = ham_stat["mean_len_msg"]
    median_len_ham = ham_stat["median_len_msg"]
    std_len_ham = ham_stat["std_len_msg"]

    # Build summary table
    tbl_email_summary = pd.DataFrame({
        "Total number of emails": [len(email)],
        "Email classification categories": ["Spam & Ham"],
        '% "Spam" emails': [f"{pct_spam * 100:.2f}% ({n_spam})"],
        '% "Ham" emails': [f"{pct_ham * 100:.2f}% ({n_ham})"],
        'Average "Spam" email length (characters)': [f"{mean_len_spam:.2f}"],
        'Median "Spam" email length (characters)': [f"{median_len_spam:.2f}"],
        'Standard Deviation "Spam" email length (characters)': [f"{std_len_spam:.2f}"],
        'Average "Ham" email length (characters)': [f"{mean_len_ham:.2f}"],
        'Median "Ham" email length (characters)': [f"{median_len_ham:.2f}"],
        'Standard Deviation "Ham" email length (characters)': [f"{std_len_ham:.2f}"]
    }).T

    tbl_email_summary.columns = ["Informations"]

    return tbl_email_summary

def clean_texts(text: str) -> str:
    """
    Clean and normalize a text string for NLP tasks.

    This function performs the following steps:
    1. Checks if the input is a string; returns an empty string if not.
    2. Converts text to lowercase and strips leading/trailing spaces.
    3. Removes URLs, email addresses, and punctuation using precompiled regex patterns.
    4. Collapses multiple whitespaces into a single space.
    5. Removes stopwords from the text.
    6. Lemmatizes the remaining words to their base form.
    7. Returns the cleaned and normalized text as a single string.

    Parameters
    ----------
    text : str
        The raw text string to clean.

    Returns
    -------
    str
        The cleaned, tokenized, stopword-free, and lemmatized text.
    """
    
    if not isinstance(text, str):
        return ""

    # Normalize case and remove leading/trailing spaces
    text = text.lower().strip()

    # Remove URLs, emails, and punctuation
    text = url_pattern.sub(" ", text)
    text = email_pattern.sub(" ", text)
    text = punc_pattern.sub(" ", text)
    # Collapse multiple whitespaces into a single space
    text = " ".join(text.split())

    # Remove stopwords
    words = [w for w in text.split() if w not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


def generate_wordcloud_message(data: pd.DataFrame, text_col: str, cat_col: str):
    """
    Generate and display word clouds for each category in a dataset.

    This function creates a word cloud for each unique category in the dataset,
    using the text from the specified column. It displays the word clouds side by side
    for visual comparison.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the emails or texts.
    text_col : str
        The name of the column containing the text/messages to visualize.
    cat_col : str
        The name of the column containing the category labels (e.g., 'spam' or 'ham').

    Returns
    -------
    None
        Displays the word clouds using matplotlib.
    """
    
    categories = data[cat_col].unique().tolist()
    n = len(categories)
    
    fig, axes = plt.subplots(1, max(n, 1), figsize=(16, 9))
    # If there is only one category, wrap the axes in a list
    if n == 1:
        axes = [axes]
    
    # Generate a word cloud for each category
    for ax, cat in zip(axes, categories):
        text = " ".join(data[data[cat_col] == cat][text_col].dropna().tolist())
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"Emails Annotated as {cat.capitalize()}", fontsize=12)
    
    plt.tight_layout(pad=4)
    plt.show()

