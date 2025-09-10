# Spam Email Detector

This project aims to create an artificial intelligence-based spam detector capable of predicting whether an email is **spam** or **ham** (non-spam). It combines techniques of **text preprocessing**, *vectorization** and **machine learning modeling** to provide a simple and effective solution for analyzing emails.

---

## Why this project is useful ?

This type of project is particularly interesting today because unwanted emails are omnipresent and can have negative consequences for individuals and professionals:  

- **For individuals**: reduce the time spent sorting emails and avoid opening potentially dangerous emails (phishing, viruses).  
- **For businesses**: protect employees and information systems, improve productivity by automatically filtering spam and reduce email security risks.  

In summary, this project proposes a practical and automated solution to improve email management and communication security.

---

## Main steps of the project

### 1. **Loading and cleaning of data**  
   - Emails are imported from a CSV file.  
   - Columns are normalized and irrelevant categories are filtered out.  

### 2. **Preprocessing and standardization of texts**  
   - Conversion of texts into lowercase letters.  
   - Removal of punctuation, URLs, email addresses and special characters.  
   - Removal of stopwords and lemmatization using NLTK.  

### 3. **Email exploration and analysis**  
   - Calculation of statistics on message length, number of messages per category, etc.  
   - Visualization of distributions and creation of wordclouds for each category.  

### 4. **Modeling and binary classification**  
   - Emails are transformed into vectors with a `CountVectorizer`.  
   - Several models are trained and compared:  
     - **Logistic Regression**  
     - **Linear SVC (SVM)**  
     - **Stochastic Gradient Descent**  
     - **Random Forest**  

### 5. **Evaluation of the models**  
   - Performances are measured with classic metrics (precision, recall, f1-score).  
   - A comparison table allows you to choose the most efficient model for the API.  

### 6. **Backup and deployment**  
   - The best model is saved in `pickle` format to be used in an API or a Streamlit application.  
   - A simple user interface allows anyone to enter the content of an email and immediately get a prediction.

---

## Limitations and precautions

Although this project is functional, it has significant limitations:  

- **Dependency on training dataset** : the model learns only from emails in the corpus. If it is not representative, generalization may be limited.  
- **Spam evolution** : spam techniques are constantly changing, which can make the template obsolete over time.  
- **Data bias** : an imbalance between categories can affect the accuracy of the model.  
- **Language complexity** : some spams use intentional mistakes, special characters or hidden links, which can be difficult to detect for a simple template.  

This project is therefore **a proof of concept** aimed at demonstrating the feasibility of a spam detector. We are aware of its constraints and limitations, and it is mainly intended for educational or demonstration purposes.

---

## Technologies and libraries used

- **Python** (3.12)  
- **Pandas** for data manipulation  
- **NLTK** for text preprocessing  
- **Scikit-learn** for vectorization and classification  
- **Matplotlib / WordCloud** for the visualizations  

---

## Local Installation

### 1. Clone the repository
```bash
# Clone the repo
git clone https://github.com/khedidata/spam-email-detection.git

# Go to the project folder
cd spam-email-detection
```

### 2. Create & Activate Virtual Environnement using [uv](https://github.com/astral-sh/uv)
```bash
# Check versions
python --version
pip install uv
uv --version

# Create virtual environment in .venv
uv venv .venv

# Activate environment
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows PowerShell
```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
```


