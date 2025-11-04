# -------------------------------------------------------------
# Modelo 3: RAG con MMR
# -------------------------------------------------------------


# Importar librerías y paquetes necesarios

from typing import List, Dict
import tiktoken
from pypdf import PdfReader  
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path
import pyarrow


# Cargar variables de entorno para Azure OpenAI

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

client = AzureOpenAI(
    api_key = AZURE_API_KEY, 
    azure_endpoint = AZURE_ENDPOINT, 
    api_version = "2024-02-01"
)

# Funciones de Lectura de Documentos

# Leer txt

def leer_txt(path: str) -> str:

    with open(path, "r", encoding = "utf-8", errors = "ignore") as f:

        return f.read()

# Leer pdf

def leer_pdf(path: str) -> str:

    lector = PdfReader(path)
    textos = []

    for pag in lector.pages:

        try:

            textos.append(pag.extract_text() or "")

        except Exception:

            textos.append("")

    return "\n".join(textos)

# Extraer por extensión

def extraer_por_extension(path: str) -> str:

    ext = Path(path).suffix.lower()

    if ext == ".pdf":

        return leer_pdf(path)

    elif ext in [".txt", ".md"]:

        return leer_txt(path)

    else:

        return ""


# Hacer chunks de los textos

def chunk_text(text: str, chunk_size: int, overlap: int,
               model_name: str = "text-embedding-3-small") -> List[str]:

    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)

    chunks = []
    desplazamiento = max(1, chunk_size - overlap)

    for inicio in range(0, len(tokens), desplazamiento):

        fin = inicio + chunk_size
        chunk_tokens = tokens[inicio:fin]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        if fin >= len(tokens):

            break

    return chunks


# Recolección de archivos

def recoleccion_archivos(input_dir: str) -> List[str]:

    exts = ["**/*.pdf", "**/*.txt"]
    archivos = []

    for pat in exts:
        
        archivos.extend(glob.glob(str(Path(input_dir) / pat), recursive = True))

    archivos = sorted(list(set(archivos)))

    return archivos


# Generar Embeddings con OpenAI

def obtener_embedding(text: str, model_name: str = "text-embedding-3-small") -> List[float]:

    response = client.embeddings.create(
        model = model_name,
        input = text
    )

    return response.data[0].embedding


# Funciones de Similitud Coseno

def similitud_coseno_manual(vec1: np.ndarray, vec2: np.ndarray) -> float:

    numerador = np.dot(vec1, vec2)
    denominador = (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    if denominador == 0:

        return 0.0
    
    return numerador / denominador


# Matriz de Similitud Coseno

def matriz_similitud_coseno(vectors: np.ndarray) -> np.ndarray:

    n = vectors.shape[0]
    matriz = np.zeros((n, n))

    for i in range(n):

        for j in range(n):

            matriz[i, j] = similitud_coseno_manual(vectors[i], vectors[j])

    return matriz


# MMR (Maximal Marginal Relevance)

def mmr(query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        doc_texts: List[str],
        relevancia: float = 0.6,
        redundancia: float = 0.4,
        k: int = 5) -> List[str]:

    # Similitud entre query y documentos

    sim_query_docs = np.array([
        similitud_coseno_manual(query_embedding, doc_embeddings[i]) for i in range(len(doc_embeddings))
    ])

    # Similitud entre documentos

    sim_docs_docs = matriz_similitud_coseno(doc_embeddings)

    seleccionados = []
    disponibles = list(range(len(doc_embeddings)))

    primero = np.argmax(sim_query_docs)
    seleccionados.append(primero)
    disponibles.remove(primero)

    # Ciclo de Iteraciones para seleccionar documentos

    while len(seleccionados) < min(k, len(doc_embeddings)):

        mmr_scores = []

        for idx in disponibles:

            relev = sim_query_docs[idx]
            redund = max(sim_docs_docs[idx][seleccionados])
            score = relevancia * relev - redundancia * redund
            mmr_scores.append(score)

        # Aca se debe de seleccionar el siguiente documento con el mayor score MMR

        siguiente = disponibles[np.argmax(mmr_scores)]
        seleccionados.append(siguiente)
        disponibles.remove(siguiente)

    return [doc_texts[i] for i in seleccionados]


# Aplicar MMR 

def aplicar_mmr(query: str,
                df_chunks: pd.DataFrame,
                index: faiss.IndexFlatL2,
                model_name: str = "text-embedding-3-small",
                k: int = 10,
                top_mmr: int = 5) -> List[str]:

    # Embedding de la consulta

    query_emb = obtener_embedding(query, model_name)
    query_emb_np = np.array(query_emb, dtype = "float32").reshape(1, -1)

    # Recuperación inicial FAISS

    distancias, indices = index.search(query_emb_np, k)

    # Extraer embeddings y textos

    embeddings_recuperados = np.vstack(df_chunks.iloc[indices[0]]["embedding"].values)
    textos_recuperados = df_chunks.iloc[indices[0]]["texto"].tolist()

    # Aplicar MMR con 0.6 de relevancia y 0.4 de redundancia

    return mmr(
        query_embedding = np.array(query_emb),
        doc_embeddings = embeddings_recuperados,
        doc_texts = textos_recuperados,
        relevancia = 0.6,
        redundancia = 0.4,
        k = top_mmr
    )


# Flujo del preprocesamiento

def main():

    input_dir = "../Datos/Documentos"
    chunk_size = 150
    overlap = 30
    model_name = "text-embedding-3-small"

    files = recoleccion_archivos(input_dir)

    print(len(files), " archivos encontrados")

    chunks_completos = []

    for f in files:

        text = extraer_por_extension(f)
        chunks = chunk_text(text, chunk_size, overlap, model_name)

        for i, chunk in enumerate(chunks):

            emb = obtener_embedding(chunk, model_name)
            chunks_completos.append({
                "documento": f,
                "chunk_id": i,
                "texto": chunk,
                "embedding": emb
            })

"""
    # Guardar chunks

    Path("Datos_Modelos/Chunks").mkdir(parents = True, exist_ok = True)
    df_chunks = pd.DataFrame(chunks_completos)
    df_chunks.to_parquet("Datos_Modelos/Chunks/chunks_con_overlapping.parquet", index = False)
    print(f"Guardados {len(df_chunks)} chunks.")

    # Crear índice FAISS

    embeddings = np.array(df_chunks["embedding"].tolist(), dtype = "float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    Path("Datos_Modelos/Indices").mkdir(parents = True, exist_ok = True)
    faiss.write_index(index, "Datos_Modelos/Indices/index_con_overlapping.faiss")
    print("Índice FAISS creado.")
"""

if __name__ == "__main__":
    main()
