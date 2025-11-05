# -------------------------------------------------------------
# Evaluación del rendimiento del RAG con BM25
# -------------------------------------------------------------


# Importar librerías necesarias

from typing import List
import tiktoken
from pypdf import PdfReader
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import faiss
import os
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI
import pyarrow
from rank_bm25 import BM25Okapi


# Cargar variables de entorno para Azure OpenAI

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

client = AzureOpenAI(
    api_key = AZURE_API_KEY,
    azure_endpoint = AZURE_ENDPOINT,
    api_version = "2024-02-01"
)

# Funciones para leer archivos (tanto .txt como .pdf)

def leer_txt(path):

    with open(path, "r", encoding = "utf-8") as f:

        return f.read()


def leer_pdf(path):

    lector = PdfReader(path)
    textos = []
    for pag in lector.pages:

        textos.append(pag.extract_text() or "")

    return "\n".join(textos)


def extraer_por_extension(path):

    ext = Path(path).suffix.lower()

    if ext == ".pdf":

        return leer_pdf(path)
    
    elif ext in [".txt", ".md"]:

        return leer_txt(path)
    
    return ""


# Hacer los chunks del texto

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

# Generar Embeddings con OpenAI

def obtener_embedding(text: str, model_name: str = "text-embedding-3-small") -> List[float]:

    response = client.embeddings.create(
        model = model_name,
        input = text
    )

    return response.data[0].embedding



# Evaluación del Recall@K para FAISS

def recall_at_k_faiss(queries, index, df, k = 5):

    correctos = 0

    for q in queries:

        emb = np.array(obtener_embedding(q["query"]), dtype = "float32").reshape(1, -1)
        distancia, idx = index.search(emb, k)
        documentos = [df.iloc[i]["documento"] for i in idx[0]]

        if any(q["documento_esperado"] in d for d in documentos):

            correctos += 1

    return correctos / len(queries)


# Evaluación del Recall@K para BM25

def recall_at_k_bm25(queries, bm25_index, df, k = 5):

    correctos = 0

    for q in queries:
        
        query_texto = q["query"]
        tokenized_query = query_texto.split()
        doc_scores = bm25_index.get_scores(tokenized_query)
        top_k_indices = np.argsort(doc_scores)[::-1][:k]
        documentos = [df.iloc[i]["documento"] for i in top_k_indices]

        if any(q["documento_esperado"] in d for d in documentos):

            correctos += 1

    return correctos / len(queries)



# Main para cargar datos y ejecutar evaluación

def main():

    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Datos", "Documentos"))
    chunk_size = 225
    overlap = 50

    patrones = [f"{input_dir}/**/*.pdf", f"{input_dir}/**/*.txt", f"{input_dir}/**/*.md"]
    archivos = []
    for patron in patrones:

        archivos.extend(glob.glob(patron, recursive = True))

    archivos = sorted(archivos)


    print(f"Encontrados {len(archivos)} archivos para procesar:")

    for archivo in archivos:

        print(f"- {os.path.basename(archivo)}")

    chunks = []

    for f in archivos:

        texto = extraer_por_extension(f)
        partes = chunk_text(texto, chunk_size, overlap)

        for i, ch in enumerate(partes):
            
            emb = obtener_embedding(ch)
            chunks.append({
                "documento": os.path.basename(f),
                "chunk_id": i,
                "texto": ch,
                "embedding": emb
            })

    df = pd.DataFrame(chunks)
    df.to_parquet("chunks_bm25.parquet")

    # Craer indice FAISS

    def get_embedding_dim(df_):

        for emb in df_["embedding"]:

            return len(emb)
            
    dim = get_embedding_dim(df)
    index = faiss.IndexFlatL2(dim)
    emb_matrix = np.vstack([np.array(e, dtype = "float32") for e in df["embedding"].values])
    index.add(emb_matrix.astype("float32"))
    faiss.write_index(index, "index_bm25.faiss")


    # Crear índice BM25

    corpus = df['texto'].tolist()
    tokenized_corpus = [doc.split() for doc in corpus]

    bm25_index = BM25Okapi(tokenized_corpus)
    
    print("Preprocesamiento completo.\n")


    # Conjunto de consultas o queries que se vann a evaluar con su respuesta esperada

    queries_evaluacion = [
        {
            "query": "It would be valid to say that most of the projected GDP growth in upcoming years will be driven by the productivity boost from AI technologies?", 
            "documento_esperado": "GS_US_Job_Market_Analysis.pdf"
        }, 
        {
            "query": "The massive inflows of capital into gold (and related assets) and its tremendous performance in a short period of time may be warning signs of a harsh times ahead for the economy?", 
            "documento_esperado": "GS_Investing_Insights.pdf"
        }, 
        {
            "query": "In terms of ratios and valuations, the current technolgy sector (boost by AI) is similar to the dotcom bubble of early 2000s?", 
            "documento_esperado": "GS_Are_We_In_Financial_Bubble.pdf"
        }, 
        {
            "query": "Why does the CAPEX financing is dangerous for the companies if it is done through debt?", 
            "documento_esperado": "GS_Are_We_In_Financial_Bubble.pdf"
        }, 
        {
            "query": "What tends to happen when the concentration of mega caps stocks exceed 25 percent or 30 percent of the index weight?", 
            "documento_esperado": "GS_Are_We_In_Financial_Bubble.pdf"
        }, 
        {
            "query": "How different are the assets allocation strategies between the pension funds and insurance companies around the?", 
            "documento_esperado": "GS_Investing_Insights.pdf"
        }, 
        {
            "query": "Why does the full consequences of AI on the labor market and employment rates only could be seen when a recession hits?", 
            "documento_esperado": "GS_US_Job_Market_Analysis.pdf"
        }, 
        {
            "query": "¿Qué políticas ha tomado la FED cuando se producen cambios tecnológicos abruptos que afectan al mercado laboral?", 
            "documento_esperado": "GS_US_Job_Market_Analysis.pdf"
        }, 
        {
            "query": "¿Es viable y qué tan rentable es diversificar un portafolio dando más peso a activos más pequeños que usando estrategias más tradicionales de diversificación como la Cartera 60/40 o la Cartera Satelital?", 
            "documento_esperado": "GS_Investing_Insights.pdf"
        },
        {
            "query": "La economía y las bolsas han tenido sectores de gran crecimiento y tendencias seculares en el pasado, como las Finazas y el Real Estate en los 1800s, el Transporte en los 1900s y la Tecnología en los 2000s. ¿Podríamos estar viviendo una tendencia similar con la Inteligencia Artificial en la actualidad?", 
            "documento_esperado": "GS_Are_We_In_Financial_Bubble.pdf"
        }
    ]

    # Evaluar Recall@5 para ambos

    r_faiss = recall_at_k_faiss(queries_evaluacion, index, df, k = 5)
    r_bm25 = recall_at_k_bm25(queries_evaluacion, bm25_index, df, k = 5)

    print("Recall@5 FAISS: ", round(r_faiss, 2))
    print("Recall@5 BM25: ", round(r_bm25, 2))

if __name__ == "__main__":

    main()