# Financial Sentiment Analysis

## Project Overview

Financial Sentiment Analysis (FSA) leverages Natural Language Processing (NLP) techniques to classify financial text data into sentiment categories such as **positive, negative, or neutral**. The financial market is highly sensitive to news and investor sentiment, making sentiment analysis an essential tool for understanding market dynamics, forecasting stock performance, and optimizing investment strategies.

With the growing use of social media, financial news outlets, and online commentary, vast amounts of textual data are generated daily. Automated sentiment analysis provides real-time insights, allowing investors and stakeholders to react swiftly to market-moving information. 

This project evaluates the performance of multiple machine learning and deep learning models for financial sentiment analysis, including:

- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Decision Tree**
- **BERT (Bidirectional Encoder Representations from Transformers)**
- **FinBERT (A financial-domain-specific BERT model)**

Recent studies show that transformer-based models like BERT and FinBERT outperform traditional machine learning models in sentiment classification, especially in domain-specific contexts such as finance.


## Model Implementation

### 1. Preprocessing

- **Text Cleaning:** Removing special characters, stopwords, and converting text to lowercase.
- **Tokenization:** Converting text into numerical features for machine learning models.
- **Vectorization:** Using TF-IDF or word embeddings for traditional models and tokenizing inputs for BERT-based models.

### 2. Machine Learning Models

- **SVM:** Utilizes a linear kernel for text classification.
- **Naive Bayes:** A probabilistic classifier based on Bayes' theorem.
- **Decision Tree:** A simple tree-based method for sentiment categorization.

### 3. Deep Learning Models

- **BERT:** A pre-trained Transformer-based model fine-tuned for sentiment classification.
- **FinBERT:** A variant of BERT specifically trained on financial text for better sentiment predictions in the financial domain.



## Evaluation Metrics

The models are evaluated using:

- **Accuracy**
- **Precision, Recall, and F1-score**
- **Confusion Matrix**

## Results & Analysis

- Traditional ML models (SVM, Naive Bayes, Decision Tree) perform decently but struggle with complex financial terminologies.
- BERT and FinBERT outperform ML models, with FinBERT providing the most domain-specific accuracy.
- **DistilBERT achieved the highest accuracy of 86% in our study, outperforming traditional models.**
- **FinBERT, despite being domain-specific, performed slightly below DistilBERT, potentially due to variations in dataset structure and vocabulary.**

## Future Improvements

- Experiment with additional datasets for better generalization.
- Fine-tune FinBERT further with custom financial datasets.
- Implement real-time sentiment analysis for stock market trends.
- Investigate entity-aware models to improve performance on financial text classification.
- Explore ensemble methods combining classical ML and transformer-based approaches.

## Authors

Developed by Ellie Do



