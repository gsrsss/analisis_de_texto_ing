import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# --- Page Config ---
st.set_page_config(page_title="TF-IDF Demo", layout="wide", initial_sidebar_state="collapsed")

# --- CSS Styles Removed ---

# --- Title and Description ---
st.title("TF-IDF Demo: Question & Answer")

st.write("""
Each line is treated as a **document** (it can be a sentence, a paragraph, or a longer text).  
Documents and questions must be in **English**, as the analysis is configured for that language. ✧˖°  

The application applies normalization and *stemming* so that words like *playing* and *play* are considered equivalent.
""")

# --- User Inputs ---
st.header("Enter your data 🪶─ .✦📜⊹₊ ݁.")

# Initial example in English
text_input = st.text_area(
    "Enter your documents (one per line):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

question = st.text_input("Enter a question", "Who is playing?")

# --- Processing Functions ---
# Initialize stemmer for English
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenize (words with length > 1)
    tokens = [t for t in text.split() if len(t) > 1]
    # Apply stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# --- Main Logic and Results ---
if st.button("Calculate TF-IDF and find answer", type="primary", use_container_width=True):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.warning("⚠️ Please enter at least one document.")
    else:
        st.header("📊 Analysis Results")
        try:
            # Vectorizer with stemming
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                stop_words="english",
                token_pattern=None
            )

            # Fit with documents
            X = vectorizer.fit_transform(documents)

            # --- Main Answer (Highlighted) ---
            
            # Question vector
            question_vec = vectorizer.transform([question])

            # Cosine similarity
            similarities = cosine_similarity(question_vec, X).flatten()

            # Most similar document
            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]

            st.write("### 🎯 Most Relevant Answer")
            st.write(f"**Your question:** {question}")
            st.write(f"**Most relevant document (Doc {best_idx+1}):**")
            st.info(f"{best_doc}")
            st.write(f"**Similarity score:** {best_score:.3f}")
            
            # Show matching stems
            vocab = vectorizer.get_feature_names_out()
            q_stems = tokenize_and_stem(question)
            matched = [s for s in q_stems if s in vocab and X[best_idx].toarray()[0][vectorizer.vocabulary_[s]] > 0]
            st.write("**Matching stems:**", f"`{', '.join(matched) or 'None'}`")
            
            # --- Additional Details (in tabs) ---
            tab1, tab2 = st.tabs(["All Scores", "TF-IDF Matrix"])

            with tab1:
                # Show all similarities
                st.write("### 📈 Similarity Scores (sorted)")
                sim_df = pd.DataFrame({
                    "Document": [f"Doc {i+1}" for i in range(len(documents))],
                    "Text": documents,
                    "Similarity": similarities
                })
                st.dataframe(sim_df.sort_values("Similarity", ascending=False))

            with tab2:
                # Show TF-IDF matrix
                st.write("### 🔢 TF-IDF Matrix (stems)")
                df_tfidf = pd.DataFrame(
                    X.toarray(),
                    columns=vectorizer.get_feature_names_out(),
                    index=[f"Doc {i+1}" for i in range(len(documents))]
                )
                st.dataframe(df_tfidf.round(3))
        
        except ValueError as e:
            if "empty vocabulary" in str(e):
                st.error("⚠️ Error: Could not build vocabulary. Make sure the documents are not empty or composed only of 'stop words' (common English words).")
            else:
                st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")



