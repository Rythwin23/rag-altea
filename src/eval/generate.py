from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
model_llm = os.getenv("MODEL_LLM")
reply = "Je ne trouve pas d'information pertinente."


def construct_prompt(query, context):
    """
    Construit un prompt pour le modèle OpenAI en utilisant la requête et le contexte fourni.

    Args:
        query (str): La requête de l'utilisateur.
        context (list): Une liste de documents contextuels pertinents.

    Returns:
        str: Le prompt construit pour le modèle OpenAI.
    """
    extraits = ""
    for i, doc in enumerate(context):
        extraits += f"{i+1}. {doc['texte']}\n"

    return f"""Extraits de la documentation :
{extraits}

Demande de l'utilisateur : {query}"""

def generate_response(client: OpenAI, query: str, context: list):
    """
    Génère une réponse en utilisant le modèle OpenAI avec la requête et le contexte fourni.

    Args:
        client (OpenAI): Une instance du client OpenAI.
        query (str): La requête de l'utilisateur.
        context (list): Une liste de documents contextuels pertinents.

    Returns:
        str: La réponse générée par le modèle OpenAI.
    """
    if not context:
        return reply

    if len(query.strip().split()) == 1:
        return f"[1] {context[0]['texte']}"

    prompt = construct_prompt(query, context)
    response = client.chat.completions.create(
        model=model_llm,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es l'assistant documentaire interne d'Altea Services. "
                    "Réponds en français et utilise uniquement les extraits fournis. "
                    "Si la demande est un thème ou un mot-clé, résume les informations "
                    "pertinentes des extraits comme une demande de synthèse. "
                    "Réponds brièvement en ne gardant que les faits pertinents à la demande. "
                    "Pour les règles, critères et exclusions, conserve exactement le sens "
                    "de l'extrait : ne transforme jamais une exclusion en condition "
                    "d'éligibilité et n'ajoute aucun fait. Cite chaque fait avec son numéro "
                    "entre crochets, par exemple [1]. Dès qu'un extrait contient une "
                    "information utile, réponds avec cette information. Ne dis jamais que "
                    "l'information est introuvable si un extrait pertinent la contient. "
                    "Si les extraits ne contiennent aucune information utile pour répondre, "
                    f"réponds exactement : {reply}"
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content