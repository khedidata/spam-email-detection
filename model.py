import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import clean_texts, loading_dataset


def main(
    category_col: str = "category",
    text_col: str = "message",
    clean_text_col: str = "clean_message"
) -> None:
    """
    Main function to load, preprocess, train a binary email classifier, and save the model.
    
    Steps:
    1. Load the dataset using `loading_dataset()`.
    2. Clean and normalize the email text using `clean_texts`.
    3. Convert the category column to binary (0 = Ham, 1 = Spam).
    4. Split the dataset into training and testing sets (80/20) with stratification.
    5. Train a LinearSVC classifier with a CountVectorizer in a pipeline.
    6. Save the trained pipeline (including vectorizer and classifier) to 'spam_detector.pkl'.
    
    Parameters
    ----------
    category_col : str, default "category"
        Name of the column containing email categories (spam/ham).
    text_col : str, default "message"
        Name of the column containing raw email text.
    clean_text_col : str, default "clean_message"
        Name of the column to store cleaned text.
    
    Returns
    -------
    None
        The function saves the trained pipeline to disk and does not return anything.
    """
    
    # 1. Load dataset
    data = loading_dataset()
    # 2. Preprocess and clean text
    data[clean_text_col] = data[text_col].apply(clean_texts)
    # 3. Encode category as binary
    data[category_col] = (data[category_col] == "spam").astype(int)
    
    # 4. Split features and target
    X, y = data[clean_text_col], data[category_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42, stratify=y
    )
    
    # 5. Define pipeline (Vectorizer + Classifier)
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("clf", LinearSVC(C=5, max_iter=5000, random_state=42))
    ])
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # 6. Save the trained model to disk
    with open("spam_detector.pkl", "wb") as f:
        pickle.dump(pipeline, f)
        
if __name__ == "__main__":
    main()
