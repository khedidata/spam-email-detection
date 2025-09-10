import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def binary_clf_model(
    clf_model, 
    text_vec: pd.Series, 
    target_vec: pd.Series) -> dict:
    """
    Train and evaluate a binary text classifier using a simple CountVectorizer + model pipeline.

    Parameters
    ----------
    clf_model : sklearn-like classifier
        Any binary classification model implementing `fit` and `predict` (e.g., LogisticRegression, SVM).
    text_vec : pd.Series
        Series containing text data (emails, messages, etc.).
    target_vec : pd.Series
        Series containing target labels (binary, e.g., 0=Ham, 1=Spam).

    Returns
    -------
    dict
        Classification report as a dictionary (precision, recall, f1-score, support for each class),
        suitable for further analysis or conversion into a DataFrame.
    """
    
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        text_vec,
        target_vec,
        train_size=0.8,
        random_state=42,
        stratify=target_vec
    )
    
    # Build a pipeline: CountVectorizer + classifier
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer()),
        ('binary_clf', clf_model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Return classification report as dict
    return classification_report(y_test, y_pred, output_dict=True)

def generate_post_ml_comparaison(
    clf_list: list,
    model_names: list
) -> pd.DataFrame:
    """
    Generate a comparison table of binary classification models.
    
    Creates a DataFrame with multi-level columns: the first level corresponds 
    to the class ('Ham' for 0 and 'Spam' for 1) and the second level to the metrics 
    (precision, recall, f1-score). Each row corresponds to a model.

    Parameters
    ----------
    clf_list : list of dict
        List of classification reports (as returned by `classification_report(..., output_dict=True)`).
    model_names : list of str
        List of model names corresponding to the reports.

    Returns
    -------
    pd.DataFrame
        DataFrame with multi-level columns: 
        - Level 0: 'Ham' and 'Spam'
        - Level 1: 'precision', 'recall', 'f1-score'
        Rows are indexed by model names.
    """
    
    # Checks
    assert all(isinstance(clf, dict) for clf in clf_list), \
        "Expected a list of classification report dictionaries. Check input."
    assert len(clf_list) == len(model_names), \
        "Number of models and model names must match."

    # Mapping of classes and metrics to keep
    class_map = {"0": "Ham", "1": "Spam"}
    metrics_to_keep = ["precision", "recall", "f1-score"]

    df_list = []
    for report_dict in clf_list:
        # Keep only classes 0 and 1
        filtered_dict = {k: v for k, v in report_dict.items() if k in class_map}

        # Create a temporary DataFrame for that model
        df = pd.DataFrame({
            (class_map[cls], metric): [filtered_dict[cls][metric]] 
            for cls in filtered_dict
            for metric in metrics_to_keep
        })
        df_list.append(df)

    # Concatenate all DataFrames and index the model names
    tbl_comparaison = pd.concat(df_list, ignore_index=True)
    tbl_comparaison.index = model_names

    # Convert columns to MultiIndex
    tbl_comparaison.columns = pd.MultiIndex.from_tuples(tbl_comparaison.columns)

    return tbl_comparaison



