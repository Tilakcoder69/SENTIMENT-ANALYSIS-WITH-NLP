# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY: CODTECH IT SOLUTIONS

NAME: BODANA TILAK

INTERN ID: CT04DN1160

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

##

During my four-week internship at **CodTech IT Solutions**, under the mentorship of **Neela Santosh**, I had the opportunity to work on a real-world project titled **“Customer Review Sentiment Analysis.”** The objective of this project was to build a machine learning system capable of automatically classifying customer feedback as *Positive*, *Neutral*, or *Negative*, and to deploy it via an interactive web interface for live predictions.

The entire project was executed using **Python**, leveraging essential libraries such as **pandas** for data handling, **scikit-learn** for machine learning workflows, and **Gradio** for deploying the model with an easy-to-use user interface.

### 1. Data Loading and Preprocessing

The project commenced with loading a dataset containing customer reviews, stored in CSV format. The dataset had two main columns: one containing the review text and the other specifying the sentiment label. Using **pandas**, the dataset was read into a DataFrame, and the review texts were extracted as features (`X`), while their corresponding sentiments served as labels (`y`). Basic preprocessing steps were applied, including converting text to lowercase and removing special characters to prepare the data for vectorization.

### 2. Text Vectorization with TF-IDF

Since machine learning algorithms require numerical inputs, the textual data had to be transformed. This was achieved through **TF-IDF (Term Frequency-Inverse Document Frequency) vectorization** using `TfidfVectorizer` from scikit-learn. This approach not only converts text into numerical values but also gives weight to words based on their frequency and importance in the corpus. Common English stopwords were removed to reduce noise and improve model performance. The resulting output was a sparse matrix representing the review texts numerically.

### 3. Model Training with Logistic Regression

After vectorizing the text data, it was divided into training and testing sets in an 80:20 ratio using `train_test_split()`. For the classification task, **Logistic Regression** was chosen due to its simplicity and strong performance in multi-class classification problems, especially involving textual data. The model was trained using the training subset of the TF-IDF-transformed data.

### 4. Model Evaluation

Post training, the model's effectiveness was assessed on the testing data. Evaluation metrics such as **accuracy**, **precision**, **recall**, and **F1-score** were computed using `accuracy_score` and `classification_report` from `sklearn.metrics`. The model demonstrated robust classification capabilities across all three sentiment categories, with balanced performance, indicating its generalizability.

### 5. Real-Time Sentiment Prediction Function

To enhance user interaction, a prediction function named `predict_sentiment()` was defined. This function takes a raw review as input, processes it through the previously fitted TF-IDF vectorizer, and uses the trained Logistic Regression model to predict the sentiment. The result is returned as a label indicating whether the sentiment is *Positive*, *Neutral*, or *Negative*.

### 6. Deployment Using Gradio Interface

The final phase of the project involved developing a **Gradio-based web interface** to enable live predictions. The interface was designed with simplicity in mind:

* **Input Text Box**: Allows users to type or paste customer reviews.
* **Output Display**: Shows the sentiment prediction in real-time.
* **Header Section**: Contains a brief title and description explaining the purpose of the interface.

With a single command (`interface.launch()`), the application was hosted in a local browser window, providing a seamless and interactive experience for users without requiring any backend deployment setup.

---

**Conclusion:**

This internship project was a highly enriching experience. It strengthened my practical knowledge of machine learning workflows and real-time model deployment. I gained hands-on expertise in data preprocessing, model evaluation, and user interface development—all of which are critical components in building production-ready AI solutions.

##

#OUTPUT

![Image](https://github.com/user-attachments/assets/b8a8e535-fa98-49cd-8825-43ba9549ed21)
![Image](https://github.com/user-attachments/assets/cd94308b-fee7-45df-a7cd-4f4cc59b1329)
