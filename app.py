import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Cargar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Datos de ejemplo
docs = [
    "Streamlit es genial para crear aplicaciones web",
    "Los modelos de embeddings son útiles para búsquedas semánticas",
    "Puedes crear interfaces fácilmente con Streamlit"
]

# Codificar documentos
embeddings = model.encode(docs, convert_to_tensor=True)

# Título de la aplicación
st.title("Búsqueda Semántica con Streamlit")

# Entrada del usuario
query = st.text_input("Ingresa tu consulta:")

# Procesar la consulta
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_index = scores.argmax().item()
    resultado = docs[top_index]

    # Mostrar resultado
    st.write("Resultado más relevante:", resultado)

#streamlit run app.py
