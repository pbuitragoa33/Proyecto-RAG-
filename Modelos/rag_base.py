# -------------------------------------------------------------
# Modelo 1: RAG Base 
# -------------------------------------------------------------


# RAG Base sin overlapping con chunks de tamaño definido


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

client = AzureOpenAI(api_key = AZURE_API_KEY, 
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


# Flujo del preprocesamiento

def main():

    input_dir = "../Datos/Documentos"
    chunk_size = 300
    overlap = 0
    model_name = "text-embedding-3-small"

    files = recoleccion_archivos(input_dir)

    print(len(files), " archivos encontrados")  # Deben de ser 3

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

    # Guardar chunks y embeddings

    Path("Datos_Modelos/Chunks").mkdir(parents = True, exist_ok = True)
    df_chunks = pd.DataFrame(chunks_completos)
    df_chunks.to_parquet("Datos_Modelos/Chunks/chunks_rag_base.parquet", index = False)
    print(f"Guardados {len(df_chunks)} chunks en Datos_Modelos/Chunks/chunks_rag_base.parquet")

    # Crear índice FAISS

    embeddings = np.array(df_chunks["embedding"].tolist(), dtype = "float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    Path("Datos_Modelos/Indices").mkdir(parents = True, exist_ok = True)
    faiss.write_index(index, "Datos_Modelos/Indices/index_rag_base.faiss")
    print("Índice FAISS creado en Datos_Modelos/Indices/index_rag_base.faiss")

if __name__ == "__main__":

    main()