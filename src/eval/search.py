"""
Module de recherche pour interroger la collection ChromaDB avec des embeddings générés par le modèle.
"""
import os

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()

modele = SentenceTransformer(os.getenv("MODELE"))
collection = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH")).get_collection(os.getenv("collection"))

def search(query, top_k=3, service=None):
    """
    Recherche les documents les plus pertinents dans la collection ChromaDB pour une requête donnée.

    Args:
        query (str): La requête de recherche.
        top_k (int, optional): Le nombre de résultats à retourner. Par défaut à 3.
        service (str, optional): Le service pour filtrer les documents. Par défaut à None.

    Returns:
        list: Une liste de documents correspondant à la requête.
    """
    vecteur = modele.encode([query]).tolist()
    filtre = {"service": service} if service else None
    results = collection.query(
        query_embeddings=vecteur,
        n_results=top_k,
        where=filtre if filtre else None
    )
    res = []
    for texte, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        res.append({
            "texte": texte,
            "meta": meta,
            "distance": distance
        })
    return res
