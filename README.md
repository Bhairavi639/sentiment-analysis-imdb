# Sentiment Analysis on IMDB Movie Reviews

## 📁 Dataset
- **Source:** [IMDB Movie Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 reviews
- **Columns:** `review` (text), `sentiment` (positive/negative)

## 🎯 Goal
Build a machine learning model to classify movie reviews as **positive** or **negative** using natural language processing (NLP) techniques.

## 🧠 Approach
- **Data Cleaning:** Remove punctuation, convert text to lowercase, and remove stopwords using NLTK.
- **Feature Extraction:** Convert cleaned text to numeric features using TF-IDF vectorization.
- **Model:** Logistic Regression classifier.
- **Evaluation:**  
  - **Accuracy Score**  
  - **Confusion Matrix**  
  - **Classification Report** (Precision, Recall, F1-Score)

## 📊 Results
- ✅ **Accuracy:** ~89.56% on the test set.
- The model effectively distinguishes positive and negative sentiments in unseen reviews.

## 🚀 How to Run

1. Clone or download this repository.
2. Install required libraries:

    ```bash
    pip install pandas scikit-learn nltk matplotlib seaborn
    ```

3. Download NLTK stopwords (first-time only):

    ```python
    import nltk
    nltk.download('stopwords')
    ```

4. Open the notebook:

    ```bash
    jupyter notebook sentiment_analysis.ipynb
    ```

5. Run all cells in order, following the step-by-step instructions in the notebook.

---

## ✅ Libraries Used
- pandas  
- scikit-learn  
- nltk  
- matplotlib  
- seaborn

---

## 📬 Contact
For questions or feedback, feel free to reach out!
