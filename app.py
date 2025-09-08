import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import PyPDF2
import io
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="AI Resume Screening App",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }

    .stTitle {
        color: #1f4e79;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 2rem;
    }

    .category-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .confidence-bar {
        background-color: #f0f2f6;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 10px 0;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #00d4aa, #00b894);
        transition: width 0.3s ease;
    }

    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }

    .stats-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except:
        return False


# Load models with caching
@st.cache_resource
def load_models():
    try:
        classifier = pickle.load(open('encoder.pkl.pkl', 'rb'))
        tfidf_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
        return classifier, tfidf_vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'encoder.pkl.pkl' and 'tfidf.pkl' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None


def clean_resume(resume_text):
    """Enhanced resume cleaning function with better regex patterns"""
    if not resume_text:
        return ""

    # Convert to string if not already
    resume_text = str(resume_text)

    # Remove URLs
    cleanText = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ',
                       resume_text)

    # Remove email addresses
    cleanText = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', cleanText)

    # Remove phone numbers
    cleanText = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', ' ', cleanText)

    # Remove RT and cc
    cleanText = re.sub(r'RT|cc', ' ', cleanText)

    # Remove mentions (@user)
    cleanText = re.sub(r'@\S+', ' ', cleanText)

    # Remove hashtags
    cleanText = re.sub(r'#\S+', ' ', cleanText)

    # Remove special characters but keep periods for abbreviations
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,\-/:;<=>?@[\\]^_`{|}~"""), ' ', cleanText)

    # Remove non-ASCII characters
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)

    # Remove extra whitespaces
    cleanText = re.sub(r'\s+', ' ', cleanText)

    # Strip leading/trailing whitespace
    cleanText = cleanText.strip()

    return cleanText


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


def get_category_mapping():
    """Return the category mapping dictionary"""
    return {
        0: "Advocate",
        1: "Arts",
        2: "Automation Testing",
        3: "Blockchain",
        4: "Business Analyst",
        5: "Civil Engineer",
        6: "Data Science",
        7: "Database",
        8: "DevOps Engineer",
        9: "DotNet Developer",
        10: "ETL Developer",
        11: "Electrical Engineering",
        12: "HR",
        13: "Hadoop",
        14: "Health and fitness",
        15: "Java Developer",
        16: "Mechanical Engineer",
        17: "Network Security Engineer",
        18: "Operations Manager",
        19: "PMO",
        20: "Python Developer",
        21: "SAP Developer",
        22: "Sales",
        23: "Testing",
        24: "Web Designing"
    }


def get_category_description(category):
    """Get description for each category"""
    descriptions = {
        "Data Science": "Involves analyzing large datasets, building ML models, and extracting insights from data.",
        "Python Developer": "Focuses on backend development, automation, and building applications using Python.",
        "Java Developer": "Specializes in enterprise applications, web development, and software engineering with Java.",
        "Web Designing": "Creates user interfaces, websites, and focuses on user experience and visual design.",
        "DevOps Engineer": "Manages infrastructure, CI/CD pipelines, and bridges development and operations.",
        "HR": "Human Resources - manages recruitment, employee relations, and organizational development.",
        "Business Analyst": "Analyzes business processes, requirements, and bridges technical and business teams.",
        "Testing": "Focuses on quality assurance, test automation, and ensuring software reliability.",
        "Database": "Specializes in database design, management, and optimization.",
        "Network Security Engineer": "Protects networks and systems from cyber threats and vulnerabilities."
    }
    return descriptions.get(category, "Professional role in the specified domain.")


def create_confidence_visualization(probabilities, categories):
    """Create a confidence visualization"""
    # Get top 5 predictions
    top_indices = probabilities.argsort()[-5:][::-1]
    top_probs = probabilities[top_indices]
    top_categories = [categories.get(i, f"Category {i}") for i in top_indices]

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=top_probs * 100,
        y=top_categories,
        orientation='h',
        marker=dict(
            color=top_probs * 100,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence %")
        )
    ))

    fig.update_layout(
        title="Top 5 Category Predictions",
        xaxis_title="Confidence (%)",
        yaxis_title="Categories",
        height=300,
        margin=dict(l=150)
    )

    return fig


def main():
    # Download NLTK data
    download_nltk_data()

    # Load models
    classifier, tfidf = load_models()

    if classifier is None or tfidf is None:
        st.stop()

    # Header
    st.markdown("<h1 class='stTitle'>🤖 AI Resume Screening App</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #666;'>Automatically categorize resumes using advanced Machine Learning</p>",
        unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📄 Upload Resume")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=['txt', 'pdf'],
            help="Upload a resume in TXT or PDF format"
        )

        # Text area for manual input
        st.markdown("### ✏️ Or Paste Resume Text")
        manual_text = st.text_area(
            "Paste resume content here:",
            height=200,
            placeholder="Paste the resume content here..."
        )

        if uploaded_file is not None or manual_text.strip():
            resume_text = ""

            # Process uploaded file
            if uploaded_file is not None:
                try:
                    if uploaded_file.type == "application/pdf":
                        resume_text = extract_text_from_pdf(uploaded_file)
                    else:
                        # Handle text files
                        try:
                            resume_text = str(uploaded_file.read(), "utf-8")
                        except UnicodeDecodeError:
                            resume_text = str(uploaded_file.read(), "latin-1")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.stop()

            # Use manual text if no file uploaded
            if not resume_text and manual_text.strip():
                resume_text = manual_text

            if resume_text:
                # Process button
                if st.button("🔍 Analyze Resume", type="primary"):
                    with st.spinner("Analyzing resume..."):
                        # Clean the resume
                        cleaned_resume = clean_resume(resume_text)

                        if not cleaned_resume.strip():
                            st.error("The resume appears to be empty or contains no valid text.")
                            st.stop()

                        # Transform and predict
                        try:
                            input_features = tfidf.transform([cleaned_resume])
                            prediction = classifier.predict(input_features)[0]
                            prediction_proba = classifier.predict_proba(input_features)[0]

                            # Get category mapping
                            category_mapping = get_category_mapping()
                            category_name = category_mapping.get(prediction, "Unknown")
                            confidence = prediction_proba[prediction] * 100

                            # Display results
                            st.markdown("---")
                            st.markdown("## 📊 Analysis Results")

                            # Main prediction card
                            st.markdown(f"""
                            <div class="category-card">
                                <h2>🎯 Predicted Category</h2>
                                <h1>{category_name}</h1>
                                <p>Confidence: {confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Category description
                            description = get_category_description(category_name)
                            st.info(f"**About this role:** {description}")

                            # Confidence visualization
                            st.markdown("### 📈 Confidence Analysis")
                            fig = create_confidence_visualization(prediction_proba, category_mapping)
                            st.plotly_chart(fig, use_container_width=True)

                            # Resume statistics
                            st.markdown("### 📋 Resume Statistics")
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                            with col_stat1:
                                word_count = len(cleaned_resume.split())
                                st.metric("Word Count", word_count)

                            with col_stat2:
                                char_count = len(cleaned_resume)
                                st.metric("Character Count", char_count)

                            with col_stat3:
                                sentence_count = len([s for s in cleaned_resume.split('.') if s.strip()])
                                st.metric("Sentences", sentence_count)

                            with col_stat4:
                                st.metric("Confidence", f"{confidence:.1f}%")

                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")

    with col2:
        st.markdown("### ℹ️ About")
        st.info("""
        This AI-powered app uses machine learning to automatically categorize resumes into different job roles.

        **Supported Categories:**
        - Data Science
        - Python Developer  
        - Java Developer
        - Web Designer
        - DevOps Engineer
        - HR
        - Business Analyst
        - And 18+ more categories
        """)

        st.markdown("### 🔧 How it Works")
        st.markdown("""
        1. **Upload** your resume (PDF/TXT)
        2. **Text Processing** - Cleans and preprocesses the text
        3. **Feature Extraction** - Uses TF-IDF vectorization
        4. **Classification** - ML model predicts the category
        5. **Results** - Shows prediction with confidence scores
        """)

        st.markdown("### 💡 Tips for Better Results")
        st.markdown("""
        - Include relevant skills and technologies
        - Mention specific project experiences
        - Use standard resume format
        - Include education and work history
        - Avoid excessive formatting
        """)


if __name__ == '__main__':
    main()