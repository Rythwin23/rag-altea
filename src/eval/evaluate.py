"""Exécute le RAG sur les questions de ``data/eval.csv``."""

import csv
from pathlib import Path

from eval.generate import client, generate_response
from eval.search import search


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVALUATION_FILE = _PROJECT_ROOT / "data" / "eval.csv"
_TOP_K = 5


def load_questions(csv_path: Path) -> list[dict[str, str]]:
    """Lit les questions et leurs documents attendus depuis le fichier TSV."""
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required_columns = {"question", "document_attendu"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "Le fichier d'évaluation doit contenir les colonnes "
                "'question' et 'document_attendu'."
            )

        questions = []
        for line_number, row in enumerate(reader, start=2):
            question = (row.get("question") or "").strip()
            expected_document = (row.get("document_attendu") or "").strip()
            if not question:
                raise ValueError(f"Question vide à la ligne {line_number} de {csv_path}.")
            questions.append({
                "question": question,
                "document_attendu": expected_document,
            })

    return questions


def _source_filename(source: str) -> str:
    return source.replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def main() -> None:
    questions = load_questions(_EVALUATION_FILE)
    if not questions:
        raise ValueError(f"Aucune question trouvée dans {_EVALUATION_FILE}.")

    expected_count = 0
    expected_found_count = 0
    for index, row in enumerate(questions, start=1):
        question = row["question"]
        expected_document = row["document_attendu"]
        results = search(question, top_k=_TOP_K)
        answer = generate_response(client, question, results)
        sources = list(dict.fromkeys(
            _source_filename((result.get("meta") or {}).get("source", "Source inconnue"))
            for result in results
        ))

        print(f"\n[{index}/{len(questions)}] {question}")
        print(f"Réponse : {answer}")
        print(f"Documents retrouvés : {', '.join(sources) if sources else 'aucun'}")
        if expected_document.casefold() != "none":
            expected_count += 1
            found = any(
                source.casefold() == expected_document.casefold()
                for source in sources
            )
            expected_found_count += found
            status = "oui" if found else "non"
            print(f"Document attendu ({expected_document}) retrouvé : {status}")
        else:
            print("Document attendu : aucun (question hors corpus)")

    print(
        f"\nRappel des documents attendus : "
        f"{expected_found_count}/{expected_count}"
    )


if __name__ == "__main__":
    main()