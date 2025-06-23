# Sentiment Analysis Project

A machine learning application that classifies text sentiment as Positive, Negative, or Neutral using multiple ML models with a modern GUI interface.

## 👥 Team Members

- **Maysoon Helmy**
- **Rana Ahmed**
- **Lamiaa Mahmoud**
- **Ahmed Hussien**
- **Hossam Abdelrahman**

## Features

- **3 ML Models**: Naive Bayes, SVM, and Logistic Regression
- **Modern GUI**: Real-time analysis with animated progress bars
- **Text Preprocessing**: NLP pipeline with tokenization, stopword removal, and lemmatization
- **TF-IDF Features**: Bigram vectorization for better text representation
- **Performance Visualization**: Model accuracy comparison charts

## Installation

```bash
pip install pandas numpy matplotlib streamlit scikit-learn imbalanced-learn nltk joblib ttkthemes Pillow
```

## Quick Start

1. **Train Models**:
   ```bash
   python main.py
   ```

2. **Run GUI**:
   ```bash
   python sentiment_analysis_app.py
   ```

## Files

- `main.py` - Model training and evaluation
- `sentiment_analysis_app.py` - GUI application
- `sentimentdataset.csv` - Training data (required)

## Dataset Format

Your CSV should include:
- `Text` - Content to analyze
- `Sentiment (Label)` - Positive/Negative/Neutral
- `Topic` - Politics/Entertainment/Sports/Business
- `Source` - Twitter/Facebook/Instagram

## GUI Features

- **Analyzer Tab**: Select model, input text, get color-coded results
- **Statistics Tab**: View model performance metrics
- **Help Tab**: Usage instructions

## Models Performance

The application trains and compares three models:
- Multinomial Naive Bayes (fast)
- SVM with linear kernel (accurate)
- Logistic Regression (balanced)

## Technical Details

- **Preprocessing**: Lowercase, punctuation removal, stopwords, lemmatization
- **Features**: TF-IDF with 2000 max features and bigrams
- **Data Split**: 80% train, 10% validation, 10% test
- **Class Balancing**: RandomOverSampler for imbalanced data

## Troubleshooting

- Run `main.py` first to generate model files
- Ensure `sentimentdataset.csv` is in the project directory
- NLTK data downloads automatically on first run

---

## 📄 License

This project was developed as part of an AI course at Ain Shams University for educational purposes.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

## 📧 Contact

For questions or suggestions, please reach out 
