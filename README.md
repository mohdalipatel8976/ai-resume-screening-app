```markdown
# 📄 AI Resume Screening App  

An end-to-end machine learning project that classifies resumes into job categories using **Natural Language Processing (NLP)** and **Machine Learning**.  
This app automates resume screening and demonstrates my ability to combine **data preprocessing, model building, and deployment with Streamlit** into a real, working product.  

---

## 🚀 Features  
- **Resume Upload:** Supports PDF and TXT formats.  
- **Text Cleaning & Preprocessing:** Removes emails, phone numbers, links, and special characters.  
- **TF-IDF Vectorization:** Converts resume text into numerical features.  
- **Machine Learning Classifier:** Predicts the candidate’s category with confidence scores.  
- **Interactive Dashboard:** Built with **Streamlit** and **Plotly** for visualization.  
- **Role Insights:** Provides a description of the predicted job category.  
- **Resume Stats:** Displays word count, sentence count, and other useful metrics.  

---

## 🧑‍💻 Tech Stack  
- **Programming Language:** Python  
- **Libraries:** Scikit-learn, Pandas, NumPy, NLTK, PyPDF2, Plotly, Streamlit  
- **Machine Learning:**  
  - TF-IDF vectorizer for feature extraction  
  - KNN Classifier wrapped in One-Vs-Rest strategy  
- **Deployment/UI:** Streamlit  

---

## 📊 Dataset  
- **Source:** A dataset of ~950 resumes labeled across 24 categories (e.g., *Data Science, Python Developer, Java Developer, DevOps Engineer, HR, Business Analyst, etc.*).  
- Performed **exploratory data analysis (EDA)** to understand category distribution and imbalance.  
- Applied preprocessing to clean raw resume text for training the model.  

---

## 🏗️ Workflow  
1. **Data Preprocessing**  
   - Cleaned text (removed links, emails, stopwords, special characters).  
   - Handled imbalanced data across categories.  

2. **Feature Engineering**  
   - Applied **TF-IDF vectorization** for text representation.  

3. **Model Training**  
   - Trained a **KNN classifier** with One-Vs-Rest strategy.  
   - Achieved **98.4% accuracy** on the test set.  

4. **Model Deployment**  
   - Saved trained model and vectorizer as `encoder.pkl.pkl` and `tfidf.pkl`.  
   - Integrated into a **Streamlit web app** with interactive UI.  

---

## 📂 Project Structure  
```

📦 AI-Resume-Screening
│── app.py                 # Streamlit app
│── encoder.pkl.pkl         # Trained ML model
│── tfidf.pkl               # TF-IDF vectorizer
│── data/                   # Dataset (optional if sharing)
│── assets/                 # Screenshots for README
│── README.md               # Project Documentation

````

---

## ⚙️ Installation  

1. Clone the repository  
```bash
git clone https://github.com/your-username/ai-resume-screening.git
cd ai-resume-screening
````

2. Create virtual environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
pip install -r requirements.txt
```

3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🎯 Results

* **Accuracy:** 98.4% on validation data
* **Top 5 Predictions:** Confidence visualization helps understand alternative role fits.
* **Practical Value:** Can reduce initial resume screening time significantly.

---

## 🌟 Learnings

* Hands-on experience in **text cleaning and preprocessing**.
* Applying **TF-IDF and machine learning classifiers** for NLP tasks.
* Building a **full-stack data science project** (EDA → Model Training → Deployment).
* Creating a **user-friendly web application** with Streamlit.

---

## 📬 Contact

👤 Mohammed Ali Patel
📧 [mohdalipatel8976@gmail.com](mailto:mohdalipatel8976@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/alipatel786)
💻 [GitHub](https://github.com/mohdalipatel8976)

---

👉 This project represents my ability to take a problem from **raw data to deployment**, blending **data science** and **software development** to build something practical and impactful.

```


