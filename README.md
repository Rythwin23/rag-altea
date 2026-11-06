# Installation et génération de la base vectorielle

## 1. Installer l'environnement

Prérequis : [uv](https://docs.astral.sh/uv/getting-started/installation/) et VS Code avec les extensions Python et Jupyter. Depuis PowerShell, place-toi à la racine du projet, puis crée l'environnement Python 3.12 et installe les dépendances :

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install -e .
```

Le fichier `requirements.txt` contient les dépendances du projet ; l'installation éditable rend le package `eval` importable depuis l'environnement `.venv`.

## 2. Générer la base avec le notebook

1. Dans VS Code, ouvre [notebook/exo01.ipynb](./notebook/exo01.ipynb).
2. Sélectionne comme noyau Jupyter l'interpréteur Python de `.venv`.
3. Vérifie que le répertoire de travail du noyau est `notebook` : exécute `import os; print(os.getcwd())` dans une cellule si nécessaire. Les chemins relatifs du notebook sont basés sur ce dossier.
4. Exécute toutes les cellules dans l'ordre. Le notebook lit `data/docs`, découpe les documents, calcule leurs embeddings et crée la collection `altea` dans `notebook/baseVectorielle`. Le modèle d'embeddings peut être téléchargé lors de la première exécution.

La sortie de la cellule d'indexation indique le nombre d'éléments dans la collection. Attention : les limites de réindexation décrites plus bas s'appliquent si la base existe déjà.

## 3. Configurer l'application

Crée un fichier `.env` à la racine avec les valeurs suivantes pour utiliser la base générée :

```dotenv
MODELE=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHROMA_DB_PATH=notebook\baseVectorielle
collection=altea
MODEL_LLM=qwen3:0.6b
```

Le chemin de la base est relatif au répertoire depuis lequel l'application est lancée. Pour générer les réponses, installe et démarre [Ollama](https://ollama.com/), puis télécharge le modèle configuré avec `ollama pull qwen3:0.6b`.

## 4. Lancer Streamlit

Depuis la racine du projet, avec la base générée, le fichier `.env` configuré et Ollama démarré, lance :

```powershell
.\.venv\Scripts\streamlit.exe run src\eval\app.py
```

L'adresse de l'interface s'affiche dans le terminal. Ouvre-la dans ton navigateur si nécessaire.

# Réponses — Préparation et indexation du corpus

Ce document rassemble les principales explications et les résultats enregistrés dans le [notebook de préparation du corpus](./notebook/exo01.ipynb). Les chiffres correspondent à son exécution enregistrée et peuvent changer si le corpus ou le nettoyage évolue.

## Collecte et nettoyage

Le corpus fictif est organisé par service dans `data/docs` : `General`, `IT` et `RH`. Lors de l'exécution enregistrée, le notebook a chargé 15 fichiers : 8 PDF et 7 fichiers texte, soit 16 pages PDF et 7 documents texte.

Les fichiers sont chargés avec `DirectoryLoader` ; les PDF utilisent `PyPDFLoader`. Chaque document ou page conserve notamment sa source, son service et son numéro de page. Ces métadonnées permettent d'afficher les sources dans les réponses et de filtrer la recherche par service.

Le nettoyage met le texte en minuscules, retire les espaces aux extrémités, les lignes vides, certains en-têtes et pieds de page répétés, ainsi que les mentions de type `page 1`. Il conserve les accents. Ce nettoyage reste imparfait : certains numéros de page isolés sont conservés et ont produit 12 chunks contenant uniquement `1` ou `2` dans la base vérifiée.

## Découpage du corpus

Le notebook utilise `RecursiveCharacterTextSplitter` pour découper chaque document ou page en morceaux appelés _chunks_, tout en conservant ses métadonnées.

## Comparaison des tailles de chunks

Le tableau produit par le notebook compare trois tailles, en caractères, avec un chevauchement de 10 %.

| Taille du chunk | Chevauchement | Nombre de chunks | Longueur moyenne |
| --------------: | ------------: | ---------------: | ---------------: |
|             200 |            20 |              183 |           140.17 |
|             500 |            50 |               82 |           310.15 |
|            1000 |           100 |               46 |           557.89 |

## Risques et choix de la taille

Un chunk trop petit peut séparer des informations qui doivent être comprises ensemble et rendre la recherche moins pertinente. Un chunk trop grand peut contenir plusieurs sujets, ce qui dilue les informations utiles et augmente la taille du contexte transmis au modèle.

Le chevauchement conserve une partie du texte entre deux chunks consécutifs. Il réduit le risque de perdre le contexte lorsqu’une phrase ou une information se trouve à la frontière entre deux chunks.

Pour la suite, le notebook retient des chunks de 750 caractères avec un chevauchement de 75 caractères. L'exécution enregistrée a produit 62 chunks. C’est un compromis à valider avec les résultats de recherche : la comparaison des tailles ne mesure pas à elle seule leur pertinence.

## Recherche lexicale avec BM25

Le notebook teste `BM25Okapi` sur les chunks. Il nettoie la question, découpe questions et chunks sur les espaces, puis classe les extraits selon leur score ; si aucun score n'est positif, il ne renvoie aucun résultat. Il teste notamment des questions sur le télétravail, les congés, les frais, le VPN et les horaires.

Cette recherche est distincte de la recherche vectorielle utilisée par l'application. Comme la séparation BM25 se fait sur les espaces, les accents et la ponctuation peuvent influencer les correspondances exactes.

## Embeddings et stockage

Le modèle `paraphrase-multilingual-MiniLM-L12-v2` produit des vecteurs de 384 valeurs.

Pour `N` chunks, il faut stocker `N × 384` valeurs numériques pour les vecteurs, sans compter les textes, métadonnées et structures de la base. Pour les 62 chunks de l'exécution enregistrée, cela représente **23 808 valeurs**.

## Indexation ChromaDB

Le notebook calcule les embeddings des chunks, puis les stocke avec leurs textes et métadonnées dans la collection persistante `altea`. `upsert` ajoute les identifiants absents et met à jour ceux qui existent déjà. Pour un corpus inchangé et les mêmes identifiants, une réexécution ne duplique pas ces entrées.

Si un document est modifié, il faut réindexer ses chunks et supprimer ses anciens chunks devenus obsolètes, notamment si leur nombre a changé. La fonction de mise à jour du notebook effectue cette suppression pour une source.

**Limite actuelle :** l'indexation complète utilise des identifiants numériques, tandis que la fonction de mise à jour par document utilise des identifiants construits à partir de la source. Une réindexation complète après une mise à jour peut donc recréer des doublons ; elle ne supprime pas non plus les anciens identifiants si le corpus rétrécit. Il faudrait utiliser une stratégie d'identifiants cohérente et nettoyer les entrées obsolètes.

## Exemple de recherche vectorielle

Dans l'exemple enregistré, la question « Combien de télétravail par semaine ? » est vectorisée avec le même modèle que les documents, puis les trois chunks les plus proches sont recherchés dans ChromaDB. Ils proviennent du document sur le télétravail ; l'un indique notamment une limite de deux jours par semaine et un autre précise que le mardi est exclu.

La recherche retourne des extraits et leurs métadonnées, pas une garantie que la réponse générée sera correcte. L'application utilise les deux premiers résultats, tandis que le script d'évaluation en utilise cinq.

## Dossier de la base vectorielle

Le dossier persistant de ChromaDB est généré localement ; l’envoyer sur GitHub ajouterait des fichiers de données volumineux et liés à l’index local. Il est ignoré par Git. Un collègue peut le recréer en installant les dépendances et en exécutant les cellules de préparation et d'indexation du notebook dans l'ordre.

Le notebook utilise le chemin relatif `baseVectorielle`, résolu depuis le répertoire de travail du noyau. La recherche de l'application utilise `CHROMA_DB_PATH` et `collection` définis dans `.env` ; ces paramètres doivent désigner la même base et la même collection que le notebook. Les chemins relatifs dépendent du répertoire depuis lequel chaque processus est lancé.

## Utiliser l'application Streamlit

Dans l'onglet **Ajouter au RAG**, sélectionne un ou plusieurs fichiers `.txt`, `.md`, `.pdf` ou `.docx`, choisis le service associé, puis clique sur **Indexer les documents**. Les fichiers sont découpés en extraits et ajoutés à la collection ChromaDB configurée pour la recherche. Réindexer un fichier portant le même nom remplace ses anciens extraits. Les nouveaux documents sont ensuite interrogeables dans l'onglet **Poser une question**.

## Évaluer les questions du jeu de test

Depuis la racine du dépôt et avec l'environnement Python du projet activé, lance `python -m eval.evaluate`. Le script lit les questions et les documents attendus dans `data/eval.csv`, interroge le RAG avec les cinq meilleurs résultats, affiche les réponses et indique si le document attendu a été retrouvé.
