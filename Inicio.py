import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# --- Configuración de la Página ---
st.set_page_config(page_title="Demo TF-IDF", layout="wide", initial_sidebar_state="collapsed")

# --- Estilos CSS ---
st.markdown("""
<style>
/* Contenedor para introducción y entradas */
.container-box {
    background-color: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Contenedor para la respuesta destacada */
.highlight-box {
    background-color: #e6f7ff; /* Azul claro */
    border: 1px solid #b3e0ff; /* Borde azul */
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}
.highlight-box h3 {
    color: #0056b3; /* Azul oscuro */
    border-bottom: 2px solid #b3e0ff;
    padding-bottom: 5px;
}

/* Contenedor para otros resultados */
.results-box {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 25px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# --- Título y Descripción ---
st.title("Demo de TF-IDF con Preguntas y Respuestas 🤖")

st.markdown('<div class="container-box">', unsafe_allow_html=True)
st.write("""
Cada línea se trata como un **documento** (puede ser una frase, un párrafo o un texto más largo).  
⚠️ Los documentos y las preguntas deben estar en **inglés**, ya que el análisis está configurado para ese idioma.  

La aplicación aplica normalización y *stemming* para que palabras como *playing* y *play* se consideren equivalentes.
""")
st.markdown('</div>', unsafe_allow_html=True)


# --- Entradas de Usuario ---
st.markdown('<div class="container-box">', unsafe_allow_html=True)
st.header("📝 Ingresa tus datos")

# Ejemplo inicial en inglés
text_input = st.text_area(
    "Escribe tus documentos (uno por línea, en inglés):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

question = st.text_input("Escribe una pregunta (en inglés):", "Who is playing?")
st.markdown('</div>', unsafe_allow_html=True)


# --- Funciones de Procesamiento ---
# Inicializar stemmer para inglés
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    # Pasar a minúsculas
    text = text.lower()
    # Eliminar caracteres no alfabéticos
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenizar (palabras con longitud > 1)
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# --- Lógica Principal y Resultados ---
if st.button("Calcular TF-IDF y buscar respuesta", type="primary", use_container_width=True):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento.")
    else:
        st.header("📊 Resultados del Análisis")
        try:
            # Vectorizador con stemming
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                stop_words="english",
                token_pattern=None
            )

            # Ajustar con documentos
            X = vectorizer.fit_transform(documents)

            # --- Respuesta Principal (Destacada) ---
            
            # Vector de la pregunta
            question_vec = vectorizer.transform([question])

            # Similitud coseno
            similarities = cosine_similarity(question_vec, X).flatten()

            # Documento más parecido
            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]

            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            st.write("### 🎯 Respuesta Más Relevante")
            st.write(f"**Tu pregunta:** {question}")
            st.write(f"**Documento más relevante (Doc {best_idx+1}):**")
            st.info(f"{best_doc}")
            st.write(f"**Puntaje de similitud:** {best_score:.3f}")
            
            # Mostrar coincidencias de stems
            vocab = vectorizer.get_feature_names_out()
            q_stems = tokenize_and_stem(question)
            matched = [s for s in q_stems if s in vocab and X[best_idx].toarray()[0][vectorizer.vocabulary_[s]] > 0]
            st.write("**Stems coincidentes:**", f"`{', '.join(matched) or 'Ninguno'}`")
            st.markdown('</div>', unsafe_allow_html=True)
            
            
            # --- Detalles Adicionales (en pestañas) ---
            tab1, tab2 = st.tabs(["Todos los Puntajes", "Matriz TF-IDF"])

            with tab1:
                # Mostrar todas las similitudes
                st.markdown('<div class="results-box">', unsafe_allow_html=True)
                st.write("### 📈 Puntajes de similitud (ordenados)")
                sim_df = pd.DataFrame({
                    "Documento": [f"Doc {i+1}" for i in range(len(documents))],
                    "Texto": documents,
                    "Similitud": similarities
                })
                st.dataframe(sim_df.sort_values("Similitud", ascending=False))
                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                # Mostrar matriz TF-IDF
                st.markdown('<div class="results-box">', unsafe_allow_html=True)
                st.write("### 🔢 Matriz TF-IDF (stems)")
                df_tfidf = pd.DataFrame(
                    X.toarray(),
                    columns=vectorizer.get_feature_names_out(),
                    index=[f"Doc {i+1}" for i in range(len(documents))]
                )
                st.dataframe(df_tfidf.round(3))
                st.markdown('</div>', unsafe_allow_html=True)
        
        except ValueError as e:
            if "empty vocabulary" in str(e):
                st.error("⚠️ Error: No se pudo construir un vocabulario. Asegúrate de que los documentos no estén vacíos o compuestos solo de 'stop words' (palabras comunes en inglés).")
            else:
                st.error(f"Ocurrió un error: {e}")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {e}")





