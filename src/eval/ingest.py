"""Ajout de fichiers à la collection ChromaDB utilisée par la recherche."""

import hashlib
from io import BytesIO
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from eval.search import collection, modele


_CHUNK_SIZE = 750
_CHUNK_OVERLAP = 75
_SOURCE_PREFIX = "Documents ajoutés"


def _extract_pages(filename: str, content: bytes) -> list[tuple[int | None, str]]:
    extension = Path(filename).suffix.lower()

    if extension in {".txt", ".md"}:
        return [(None, content.decode("utf-8-sig"))]

    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        return [
            (page_number, page.extract_text() or "")
            for page_number, page in enumerate(reader.pages, start=1)
        ]

    if extension == ".docx":
        from docx import Document

        document = Document(BytesIO(content))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        tables = [
            " | ".join(cell.text.strip() for cell in row.cells)
            for table in document.tables
            for row in table.rows
        ]
        return [(None, "\n".join(paragraphs + tables))]

    raise ValueError(f"Format non pris en charge : {extension or 'sans extension'}")


def index_uploaded_file(filename: str, content: bytes, service: str) -> int:
    """Découpe et indexe un fichier; remplace ses anciens extraits s'il a déjà été ajouté."""
    safe_filename = Path(filename).name
    source = f"{_SOURCE_PREFIX}/{safe_filename}"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )

    chunks = []
    metadatas = []
    for page_number, text in _extract_pages(safe_filename, content):
        for chunk in splitter.split_text(text):
            chunks.append(chunk)
            metadata: dict[str, str | int] = {"source": source, "service": service}
            if page_number is not None:
                metadata["page"] = page_number
            metadatas.append(metadata)

    if not chunks:
        raise ValueError("Le fichier ne contient aucun texte exploitable.")

    embeddings = modele.encode(chunks).tolist()
    ids = [
        hashlib.sha256(f"{source}\0{index}".encode("utf-8")).hexdigest()
        for index in range(len(chunks))
    ]

    anciens_ids = set(collection.get(where={"source": source})["ids"])
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    ids_obsoletes = anciens_ids - set(ids)
    if ids_obsoletes:
        collection.delete(ids=list(ids_obsoletes))

    return len(chunks)
