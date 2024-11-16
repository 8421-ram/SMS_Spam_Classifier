# SMS Spam Detection

## 📜 Project Overview

This project focuses on **SMS Spam Detection**, an essential task in Natural Language Processing (NLP) that aims to classify incoming text messages (SMS) as either spam or legitimate (ham). By leveraging machine learning algorithms, this project provides an automated solution to identify unwanted spam messages, helping reduce clutter and potential security threats.
<img src="https://res.cloudinary.com/dgwuwwqom/image/upload/v1731750437/Email_spam.jpg" alt="Description of the Image" width="600"/>


## 🚀 Key Features

- **Data Preprocessing**: Efficient data cleaning, tokenization, and text normalization (e.g., converting to lowercase, removing punctuation, and stopwords).
- **Feature Engineering**: Utilizes techniques such as Bag of Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), and Word Embeddings.
- **Machine Learning Models**: Implements multiple classifiers including:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
- **Model Evaluation**: Provides metrics like accuracy, precision, recall, F1-score, and confusion matrix to assess model performance.
- **User Interface**: A simple command-line or graphical interface for testing SMS messages in real-time.

## 📊 Dataset

The project uses the popular **SMS Spam Collection dataset**, which consists of over 5,000 SMS messages labeled as "ham" (legitimate) or "spam". The dataset includes messages in English and has been pre-labeled for training and evaluation.

- **Dataset Link**: [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Data Format**: Each line in the dataset is composed of a label ("spam" or "ham") followed by the message text.

## 🛠️ Installation

To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/sms-spam-detection.git
   cd sms-spam-detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

## 🧰 Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn
- **NLP Techniques**: Tokenization, Lemmatization, TF-IDF, BoW
- **Machine Learning Models**: Logistic Regression, Naive Bayes, SVM, Random Forest

## 📂 Project Structure

```
sms-spam-detection/
├── data/
│   └── spam.csv
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── sms_predictor.py
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

## 📊 Results & Analysis

- The models were effective in identifying spam messages, with Logistic Regression and SVM delivering reliable classification.
- Naive Bayes demonstrated efficiency in speed and accuracy, making it suitable for real-time detection.
- Feature analysis revealed that specific keywords like "free", "win", "cash", and "prize" were strong indicators of spam.



## 📧 Contact

If you have any questions or feedback, feel free to reach out:

- **Your Name**: [Your Email](mailto:your.ramagopalakrishna7818@gmail.com)
- **GitHub**: [Your GitHub Profile](https://github.com/8421-ram)
