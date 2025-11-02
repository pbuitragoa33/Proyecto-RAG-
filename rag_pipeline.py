# --------------------------------------------------------------------------------------
# Pipeline para RAG completo (Preprocesamiento, Indexación, Búsqueda y Generación)
# --------------------------------------------------------------------------------------

# Importar librerías necesarias

import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import AzureOpenAI
import pyarrow

# Cargar variables de entorno para Azure OpenAI

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

client = AzureOpenAI(api_key = AZURE_API_KEY, 
                     azure_endpoint = AZURE_ENDPOINT, 
                     api_version = "2024-02-01" 
)

# Cargar el índice FAISS y el DataFrame de los chunks

# Cargar chunks

chunks_df = pd.read_parquet("Datos/Chunks/chunks.parquet")
chunks_textos = chunks_df["texto"].tolist()

# Cargar índice FAISS

index = faiss.read_index("Datos/Indices/index.faiss")


# Función para obtener embedding de una consulta (query)

def obtener_embedding_query(query: str, model_name: str = "text-embedding-3-small") -> np.ndarray:

    respuesta = client.embeddings.create(
        input = query,
        model = model_name
    )

    embdg = np.array(respuesta.data[0].embedding, dtype = "float32")

    return embdg


# Función para buscar los chunks más similares a la consulta

def buscar_chunks_similares(query: str, k: int = 5):

    emb_query = obtener_embedding_query(query).reshape(1, -1)
    distancias, indices = index.search(emb_query, k)
    resultados = [chunks_textos[i] for i in indices[0]]

    return resultados


# Función principal - Generación de respuesta basada en RAG

def generar_respuesta_rag(query: str, top_k: int = 5):

    # Buscar chunks similares

    top_chunks = buscar_chunks_similares(query, k = top_k)

    # Concatenar los chunks encontrados como contexto

    contexto = "\n".join(top_chunks)

    # Crear el prompt para la generación de respuesta

    prompt = f"""
    Con la siguiente información extraída de documentos:{contexto}
    Responde de manera clara, completa y elaborada a la pregunta: {query}
    """

    # Generar respuesta usando el modelo de lenguaje (GPT)

    respuesta = client.chat.completions.create(
        messages = [{"role": "user", "content": prompt}],
        model = "gpt-4.1-nano",
        temperature = 0.2
    )

    return respuesta.choices[0].message.content


# Ejemplo de uso

if __name__ == "__main__":

    while True:

        query = input("Escribe la consulta (o 'salir' para terminar): ")
        
        if query.lower() in ["salir", "exit"]:

            break

        respuesta = generar_respuesta_rag(query)

        print("\nRespuesta RAG:\n", respuesta)
        print("-" * 30)


