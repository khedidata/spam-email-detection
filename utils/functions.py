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

def loading_dataset(
    data_folder_name: str = "data/",
    csv_file: str = "email.csv") -> pd.DataFrame:
    assert any(f.endswith(".csv") for f in os.listdir("data/")), \
    "Initial working file not found in data/ folder !"
    
    data = pd.read_csv(os.path.join(data_folder_name, csv_file))
    data.columns = [c.lower() for c in data.columns]
    categories_list = ['ham', 'spam']
    data = data[data["category"].isin(categories_list)]
    
    return data

def generate_email_summary(data: pd.DataFrame) -> pd.DataFrame:
    
    email = data.copy()
    email["message_len"] = email["message"].str.len()
    
    email["is_spam"] = email["category"].str.lower().eq("spam")
    email["is_ham"] = email["category"].str.lower().eq("ham")
    email["is_nan_message"] = email["category"].isna()
    
    stats = email.groupby("category").agg(
        n_msg=("message", "size"),
        n_nan=("is_nan_message", "sum"),
        mean_len_msg=("message_len", "mean"),
        median_len_msg=("message_len", "median"),
        std_len_msg=("message_len", "std")
    )
    
    n_spam = email["is_spam"].sum()
    n_ham = email["is_ham"].sum()
    pct_spam = email["is_spam"].sum() / len(email)
    pct_ham = email["is_ham"].sum() / len(email)
    
    spam_stat = stats[stats.index == "spam"]
    mean_len_spam = spam_stat["mean_len_msg"].values[0]
    median_len_spam = spam_stat["median_len_msg"].values[0]
    std_len_spam = spam_stat["std_len_msg"].values[0]
    
    ham_stat = stats[stats.index == "ham"]
    mean_len_ham = ham_stat["mean_len_msg"].values[0]
    median_len_ham = ham_stat["median_len_msg"].values[0]
    std_len_ham = ham_stat["std_len_msg"].values[0]
    
    tbl_email_summary = pd.DataFrame(
        {
            "Total number of emails" : [len(email)],
            "Email classification categories" : [f"Spam & Ham"],
            '% "Spam" emails' : [f"{pct_spam * 100:.2f}% ({n_spam})"],
            '% "Ham" emails' : [f"{pct_ham * 100:.2f}% ({n_ham})"],
            'Average "Spam" email length (characters)' : [f"{mean_len_spam:.2f}"],
            'Median "Spam" email length (characters)' : [f"{median_len_spam:.2f}"],
            'Standard Deviation "Spam" email length (characters)' : [f"{std_len_spam:.2f}"],
            'Average "Ham" email length (characters)' : [f"{mean_len_ham:.2f}"],
            'Median "Ham" email length (characters)' : [f"{median_len_ham:.2f}"],
            'Standard Deviation "Ham" email length (characters)' : [f"{std_len_ham:.2f}"]
        }
    ).T
    tbl_email_summary.columns = ["Informations"]
    
    return tbl_email_summary

def clean_texts(text: str) -> str:
    
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = url_pattern.sub(" ", text)
    text = email_pattern.sub(" ", text)
    text = punc_pattern.sub(" ", text)
    
    text = " ".join(text.split())
    
    words = [w for w in text.split() if w not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return " ".join(words)

def generate_wordcloud_message(
    data: pd.DataFrame,
    text_col: str,
    cat_col: str):
    
    categories = data[cat_col].unique().tolist()
    n = len(categories)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    
    if n == 1:
        axes = [axes]
        
    for ax, cat in zip(axes, categories):
        text = " ".join(data[data[cat_col] == cat][text_col].dropna().tolist())
        
        wordcloud = WordCloud(width=800, height=400,
                              background_color='white',
                              colormap='viridis',
                              max_words=100).generate(text)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"Emails Annotaded as {cat.capitalize()}", fontsize=12)
    plt.tight_layout(pad=4)
    plt.show()
