# -------------------------------------------------------------
# Busqueda de información utilizando embeddings y FAISS
# -------------------------------------------------------------


# Importar librerías y modulos necesarios

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

# Ejemplo de usp

if __name__ == "__main__":

    consulta = input("Ingrese su consulta: ")
    top_chunks = buscar_chunks_similares(consulta, k = 5)

    print("\nChunks más similares encontrados:\n")

    for i, chunks in enumerate(top_chunks, start = 1):

        print(f"Chunk {i}:\n{chunks}\n")

