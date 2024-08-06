# Amazon_Mobile_Reviews-for-Top-20-Brands
Amazon_Mobile_Reviews_analysis for Top 20 Brands
# Amazon Mobile Reviews for Top 20 Brands

This repository contains a project that analyzes Amazon unlocked mobile reviews for the top 20 brands. The analysis involves data cleaning, visualization, sentiment analysis, and model evaluation.

## Table of Contents
- [Installation](#installation)
- [Data Description](#data-description)
- [Data Analysis](#data-analysis)
- [Data Cleaning](#data-cleaning)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use the code in this repository, ensure you have the following libraries installed:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
beautifulsoup4
nltk
```

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn beautifulsoup4 nltk
```

## Data Description

The dataset used in this project is `Amazon_Unlocked_Mobile.csv`. It contains reviews of unlocked mobile phones sold on Amazon. The main columns of interest are:
- `Brand Name`: The brand of the mobile phone.
- `Product Name`: The name of the mobile phone.
- `Reviews`: The text of the review.
- `Rating`: The rating given by the reviewer.

## Data Analysis

The analysis involves several steps:

1. **Summary Statistics**:
    ```python
    print("Summary statistics of numerical features : \n", df.describe())
    ```

2. **Distribution of Ratings**:
    ```python
    plt.figure(figsize=(12,8))
    df['Rating'].value_counts().sort_index().plot(kind='bar')
    plt.title('Distribution of Rating')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    ```

3. **Number of Reviews for Top 20 Brands**:
    ```python
    plt.figure(figsize=(12,8))
    brands[:20].plot(kind='bar')
    plt.title("Number of Reviews for Top 20 Brands")
    ```

4. **Distribution of Review Length**:
    ```python
    plt.figure(figsize=(12,8))
    review_length.loc[review_length < 1500].hist()
    plt.title("Distribution of Review Length")
    plt.xlabel('Review length (Number of characters)')
    plt.ylabel('Count')
    ```

## Data Cleaning

The text data is cleaned by removing HTML tags, non-character symbols, stopwords, and by performing stemming. The cleaned text is then used for further analysis.

```python
def cleanText(raw_text, remove_stopwords=False, stemming=False, split_text=False):
    text = BeautifulSoup(raw_text, 'lxml').get_text()  # remove HTML
    letters_only = re.sub("[^a-zA-Z]", " ", text)  # remove non-character
    words = letters_only.lower().split()  # convert to lower case
    
    if remove_stopwords:  # remove stopwords
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        
    if stemming:  # perform stemming
        stemmer = SnowballStemmer('english') 
        words = [stemmer.stem(w) for w in words]
        
    return " ".join(words)
```

## Model Training and Evaluation

The cleaned data is split into training and testing sets. A `TfidfVectorizer` is used to convert the text data into numerical features. A Logistic Regression model is then trained and evaluated.

```python
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)

# Evaluate the model
predictions = lr.predict(tfidf.transform(X_test_cleaned))
modelEvaluation(predictions)
```

## Results

The model's performance is evaluated using accuracy, ROC-AUC score, classification report, and confusion matrix.

```python
def modelEvaluation(predictions):
    print("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("\nROC-AUC score: {:.4f}".format(roc_auc_score(y_test, predictions)))
    print("\nClassification report:\n", metrics.classification_report(y_test, predictions))
    print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, predictions))
```

## Contributing

Contributions are welcome! Please create an issue or submit a pull request for any improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
