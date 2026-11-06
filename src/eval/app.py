import streamlit as st

st.set_page_config(page_title="Altéa | Assistant documentaire", layout="centered")
st.title("Assistant documentaire Altéa")

try:
    from eval.search import search
    from eval.generate import client, generate_response
    from eval.ingest import index_uploaded_file
except Exception as error:
    st.error(f"Initialisation impossible : {error}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

questions_tab, ingestion_tab = st.tabs(["Poser une question", "Ajouter au RAG"])

with questions_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            documents = message.get("documents", [])
            if documents:
                for index, document in enumerate(documents, start=1):
                    with st.expander(f"Document pertinent {index} — {document['source']}"):
                        st.markdown(document["text"])
            else:
                for source in message.get("sources", []):
                    st.caption(source)

    question = st.chat_input("Pose ta question sur la documentation")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les documents..."):
                try:
                    sources = search(question, top_k=2)
                    answer = generate_response(client, question, sources)
                    st.markdown(answer)

                    source_labels = []
                    relevant_documents = []
                    for item in sources:
                        metadata = item.get("meta") or {}
                        label = metadata.get("source", "Source inconnue")
                        if metadata.get("page") is not None:
                            label += f", page {metadata['page']}"
                        source_labels.append(label)
                        relevant_documents.append({
                            "source": label,
                            "text": item["texte"],
                        })

                    for index, document in enumerate(relevant_documents, start=1):
                        with st.expander(f"Document pertinent {index} — {document['source']}"):
                            st.markdown(document["text"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_labels,
                        "documents": relevant_documents,
                    })
                except Exception as error:
                    st.error(f"Erreur pendant la recherche ou la génération : {error}")

with ingestion_tab:
    st.write("Ajoute des documents pour les rendre disponibles dans les prochaines recherches.")
    uploaded_files = st.file_uploader(
        "Documents à indexer",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )
    service = st.selectbox("Service associé", ["General", "RH", "IT"])

    if st.button("Indexer les documents", disabled=not uploaded_files):
        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"Indexation de {uploaded_file.name}..."):
                    chunk_count = index_uploaded_file(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        service,
                    )
                st.success(f"{uploaded_file.name} ajouté au RAG ({chunk_count} extraits).")
            except Exception as error:
                st.error(f"Impossible d'indexer {uploaded_file.name} : {error}")