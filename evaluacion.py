# --------------------------------------------------------------------------------------
# Evaluación del Rendimiento del RAG
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


# Cargar el iindice FAISS y el DataFrame de los chunks (Original)

chunks_df = pd.read_parquet("Datos/Chunks/chunks.parquet")
index = faiss.read_index("Datos/Indices/index.faiss")


# Cargar el índice FAISS y el DataFrame de los chunks (RAG Base)

#chunks_df = pd.read_parquet("Modelos/Datos_Modelos/Chunks/chunks_rag_base.parquet")
#index = faiss.read_index("Modelos/Datos_Modelos/Indices/index_rag_base.faiss")

# Cargar el índice FAISS y el DataFrame de los chunks (RAG Overlap)

#chunks_df = pd.read_parquet("Modelos/Datos_Modelos/Chunks/chunks_con_overlapping.parquet")
#index = faiss.read_index("Modelos/Datos_Modelos/Indices/index_con_overlapping.faiss")



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


# Función para calcular el Recall@K

def recall_at_k(queries, index, df, k = 5):

    correctos = 0

    for q in queries:

        emb = np.array(client.embeddings.create(
            input = q["query"],
            model = "text-embedding-3-small").data[0].embedding, dtype = "float32").reshape(1, -1)

        distancias, indices = index.search(emb, k)
        documentos = [df.iloc[i]["documento"] for i in indices[0]]

        if any(q["documento_esperado"] in doc for doc in documentos):
        
            correctos += 1
    
    recall = correctos / len(queries)

    return recall

# Calcular y mostrar el Recall@K

print("Recall@5:", recall_at_k(queries_evaluacion, index, chunks_df))


# Para correr este script

if __name__ == "__main__":
    
    recall5 = recall_at_k(queries_evaluacion, index, chunks_df, k = 5)
    print(f"\nRecall@5: {recall5:.2f}")



"""
1. Se cargan los archivos .faiss y .parquet necesarios.
2. Se crean varias consultas juntos con el documento esperado para cada una.
3. Se hace la función recall_at_k que calcula el Recall@K para las consultas dadas.
"""