import os
import streamlit as st
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Initialize research database
db = Chroma(
    collection_name="medresearch_database",
    embedding_function=embedding_model,
    persist_directory='./medresearch_db'
)


def clear_current_vector_database():
    """
    HARD RESET the Chroma collection.

    A document-by-document delete can silently fail and leave old vectors in the
    persistent collection. Those old vectors can reuse paper_id values after a
    fresh upload and cause an unselected PDF to appear in retrieval results.
    Recreating the entire collection removes that possibility.
    """
    global db

    try:
        db.delete_collection()
    except Exception as error:
        print(f"Collection reset notice: {error}")

    db = Chroma(
        collection_name="medresearch_database",
        embedding_function=embedding_model,
        persist_directory="./medresearch_db"
    )


def format_docs(docs, include_source_labels=False):
    """Format retrieved passages, optionally preserving document identity."""
    formatted = []

    for index, doc in enumerate(docs, start=1):
        if include_source_labels:
            source_name = doc.metadata.get("paper_filename")
            if not source_name:
                source_name = os.path.basename(
                    doc.metadata.get("source", "Unknown document")
                )

            paper_id = doc.metadata.get("paper_id", "unknown")
            page = doc.metadata.get("page")
            page_label = f"Page {page + 1}" if page is not None else "Page unavailable"

            formatted.append(
                f"[DOCUMENT {index} | PAPER_ID: {paper_id} | "
                f"FILE: {source_name} | {page_label}]\n"
                f"{doc.page_content}"
            )
        else:
            formatted.append(doc.page_content)

    return "\n\n".join(formatted)




def add_to_db(research_papers):
    """Processes and adds uploaded PDF files to the database.

    This function checks if any files have been uploaded. If files are uploaded,
    it saves each file to a temporary location, processes the content using a PDF loader,
    and splits the content into smaller chunks. Each chunk, along with its metadata, 
    is then added to the database. Temporary files are removed after processing.

    Args:
        research_papers (list): A list of research paper dictionaries to be processed.

    Returns:
        None"""
    # Check if files are uploaded
    if not research_papers:
        st.error("No files uploaded!")
        return



    for paper in research_papers:

        paper_id = paper["paper_id"]
        paper_filename = paper["filename"]
        uploaded_file = paper["file"]

        # Save the uploaded file to a temporary path
        temp_file_path = os.path.join("./temp", uploaded_file.name)

        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        # Load the file using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()

        # Store metadata and content
        doc_metadata = [data[i].metadata for i in range(len(data))]
        doc_content = [data[i].page_content for i in range(len(data))]


         # Remove empty pages
        valid_content = []
        valid_metadata = []

        for content, metadata in zip(
            doc_content,
            doc_metadata
        ):
            if content and content.strip():

                valid_content.append(content)
                valid_metadata.append(metadata)

        # Skip paper if no readable text was extracted
        if not valid_content:

            st.warning(
                f"⚠️ No readable text found in "
                f"'{paper_filename}'. "
                f"This paper may be scanned/image-based."
            )

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            continue

        doc_content = valid_content
        doc_metadata = valid_metadata


        for metadata in doc_metadata:
            metadata["paper_id"] = paper_id
            metadata["paper_filename"] = paper_filename


        # Split documents into smaller chunks
        st_text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",
            chunk_size=100,
            chunk_overlap=50
        )


        st_chunks = st_text_splitter.create_documents(
            doc_content,
            doc_metadata
        )

        # Safety check before adding to ChromaDB
        if not st_chunks:

            st.warning(
                f"⚠️ No text chunks were created for "
                f"'{paper_filename}'. Skipping this paper."
            )

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            continue

        # Add chunks to database
        db.add_documents(st_chunks)

        # Remove the temporary file after processing
        if os.path.exists(temp_file_path):
         os.remove(temp_file_path)







def remove_redundant_documents(
    documents,
    similarity_threshold=0.90
):
    """Remove near-duplicate retrieved documents."""

    if not documents:
        return []

    if len(documents) == 1:
        return documents

    # Extract text from retrieved documents
    texts = [
        doc.page_content.strip()
        for doc in documents
    ]

    # Create embeddings for the retrieved chunks
    embeddings = embedding_model.embed_documents(
        texts
    )

    selected_documents = []
    selected_embeddings = []

    for index, document in enumerate(documents):

        current_embedding = np.array(
            embeddings[index]
        ).reshape(1, -1)

        is_redundant = False

        for selected_embedding in selected_embeddings:

            similarity = cosine_similarity(
                current_embedding,
                selected_embedding
            )[0][0]

            if similarity >= similarity_threshold:
                is_redundant = True
                break

        if not is_redundant:

            selected_documents.append(
                document
            )

            selected_embeddings.append(
                current_embedding
            )

    return selected_documents






def identify_relevant_papers(
    query,
    research_papers,
    max_candidates=15,
    relevance_margin=0.10
):
    """
    Identify which currently processed research papers
    are relevant to the user's question.
    """

    # Only consider papers currently marked as processed.
    processed_papers = [
        paper
        for paper in research_papers
        if paper.get("processed", False)
    ]

    if not processed_papers:
        return []

    # IDs of papers that currently exist in the research library.
    current_paper_ids = {
        paper["paper_id"]
        for paper in processed_papers
    }

    # Retrieve candidate chunks together with similarity scores.
    scored_docs = db.similarity_search_with_score(
        query,
        k=max_candidates
    )

    if not scored_docs:
        return []

    # Store the strongest relevance score for each paper.
    paper_scores = {}

    for doc, distance in scored_docs:

        paper_id = doc.metadata.get("paper_id")

        # Ignore documents that are not part of the current
        # processed research-paper library.
        if paper_id not in current_paper_ids:
            continue

        # Smaller Chroma distance means greater similarity.
        relevance = 1 / (1 + distance)

        if (
            paper_id not in paper_scores
            or relevance > paper_scores[paper_id]
        ):
            paper_scores[paper_id] = relevance

    if not paper_scores:
        return []

    # Find the strongest matching paper.
    best_score = max(paper_scores.values())

    # Keep papers whose relevance is sufficiently close
    # to the strongest matching paper.
    relevant_paper_ids = [
        paper_id
        for paper_id, score in paper_scores.items()
        if score >= best_score - relevance_margin
    ]

    return relevant_paper_ids






def retrieve_comparison_documents(
    query,
    selected_paper_ids,
    max_total_documents=5
):
    """
    Retrieve comparison evidence separately from each
    explicitly selected research paper.

    Comparison retrieval focuses on both the user's question
    and important research sections such as findings, results,
    conclusions, methodology, and limitations.
    """

    if not selected_paper_ids:
        return []

    all_documents = []

    # Create a research-focused retrieval query.
    comparison_query = f"""
    {query}

    Focus on information such as:
    research objective,
    methodology,
    methods,
    experiments,
    results,
    findings,
    conclusions,
    limitations,
    and important research takeaways.
    """

    # Retrieve evidence independently from each selected paper.

    for paper_id in selected_paper_ids:

        paper_docs = db.max_marginal_relevance_search(
            comparison_query,
            k=5,
            fetch_k=20,
            lambda_mult=0.5,
            filter={
                "paper_id": paper_id
            }
        )

        all_documents.extend(paper_docs)

    if not all_documents:
        return []

    # Give every selected paper at least one opportunity
    # to contribute evidence.
    selected_documents = []
    represented_paper_ids = set()

    for document in all_documents:

        paper_id = document.metadata.get("paper_id")

        if (
            paper_id not in represented_paper_ids
            and len(selected_documents) < max_total_documents
        ):
            selected_documents.append(document)
            represented_paper_ids.add(paper_id)

    # Fill remaining slots with other relevant passages.
    for document in all_documents:

        if len(selected_documents) >= max_total_documents:
            break

        if document not in selected_documents:
            selected_documents.append(document)

    return selected_documents








def _filter_relevant_documents(query, paper_ids=None, k=5, min_relevance=0.35):
    """Retrieve with scores and reject clearly unrelated nearest-neighbour matches."""
    search_kwargs = {}
    if paper_ids:
        search_kwargs["filter"] = {"paper_id": {"$in": paper_ids}}

    scored = db.similarity_search_with_score(query, k=max(k * 4, 12), **search_kwargs)
    relevant = []
    for doc, distance in scored:
        relevance = 1 / (1 + float(distance))
        if relevance >= min_relevance:
            doc.metadata["retrieval_relevance"] = relevance
            relevant.append(doc)
        if len(relevant) >= k:
            break
    return relevant





def _balanced_normal_retrieval(
    query,
    paper_ids,
    max_total=5,
    min_relevance=0.35
):
    """
    Retrieve relevant evidence from explicitly selected papers.

    IMPORTANT:
    - Does NOT force every selected paper to contribute evidence.
    - Rejects clearly unrelated nearest-neighbour matches.
    - Keeps multiple-PDF questions working normally.
    - Returns an empty list when the question is unrelated
      to all selected research papers.
    """

    if not paper_ids:
        return []

    per_paper = []

    for paper_id in paper_ids:

        # Retrieve documents WITH similarity scores
        scored_docs = db.similarity_search_with_score(
            query,
            k=8,
            filter={
                "paper_id": paper_id
            }
        )

        relevant_docs = []

        for doc, distance in scored_docs:

            # Chroma distance:
            # smaller distance = more relevant
            relevance = 1 / (
                1 + float(distance)
            )

            # Reject unrelated chunks
            if relevance >= min_relevance:

                doc.metadata[
                    "retrieval_relevance"
                ] = relevance

                relevant_docs.append(
                    doc
                )

        # Remove duplicate passages inside
        # this paper only
        relevant_docs = remove_redundant_documents(
            relevant_docs
        )

        if relevant_docs:

            per_paper.append(
                (
                    paper_id,
                    relevant_docs
                )
            )

    # IMPORTANT:
    # If no selected paper contains genuinely relevant
    # evidence, return nothing.
    if not per_paper:
        return []

    selected = []
    seen = set()

    # First pass:
    # Take the best relevant passage from
    # every relevant paper.
    for paper_id, docs in per_paper:

        if len(selected) >= max_total:
            break

        doc = docs[0]

        key = (
            doc.metadata.get("paper_id"),
            doc.metadata.get("page"),
            doc.page_content
        )

        if key not in seen:

            selected.append(doc)

            seen.add(key)

    # Second pass:
    # Add additional relevant passages
    # fairly across papers.
    index = 1

    while len(selected) < max_total:

        added = False

        for paper_id, docs in per_paper:

            if len(selected) >= max_total:
                break

            if index < len(docs):

                doc = docs[index]

                key = (
                    doc.metadata.get("paper_id"),
                    doc.metadata.get("page"),
                    doc.page_content
                )

                if key not in seen:

                    selected.append(doc)

                    seen.add(key)

                    added = True

        if not added:
            break

        index += 1

    return selected







def run_rag_chain(query, comparison_mode=False, selected_paper_ids=None):
    """Generate an evidence-grounded answer and return only genuinely relevant evidence."""
    selected_paper_ids = selected_paper_ids or []

    # Comparison is intentionally strict: it must have at least two explicit papers.
    if comparison_mode and len(selected_paper_ids) < 2:
        return ("⚠️ Please select at least 2 research papers to use comparison mode.", [])

    if comparison_mode:
        retrieved_docs = retrieve_comparison_documents(
            query=query,
            selected_paper_ids=selected_paper_ids,
            max_total_documents=6
        )
    elif selected_paper_ids:
        # Keep explicit filtering, but avoid one paper dominating multi-paper questions.
        if len(selected_paper_ids) == 1:
            retrieved_docs = db.max_marginal_relevance_search(
                query, k=5, fetch_k=20, lambda_mult=0.5,
                filter={"paper_id": {"$in": selected_paper_ids}}
            )
        else:
            retrieved_docs = _balanced_normal_retrieval(query, selected_paper_ids, max_total=5)
    else:
        retrieval_paper_ids = identify_relevant_papers(
            query=query,
            research_papers=st.session_state.research_papers
        )
        if not retrieval_paper_ids:
            return ("The uploaded research material does not contain enough relevant information to answer this question.", [])
        retrieved_docs = db.max_marginal_relevance_search(
            query, k=5, fetch_k=20, lambda_mult=0.5,
            filter={"paper_id": {"$in": retrieval_paper_ids}}
        )

    if comparison_mode:
        # Keep comparison evidence paper-balanced. A global second search or
        # cross-paper de-duplication can silently remove one selected paper.
        represented_ids = {
            doc.metadata.get("paper_id")
            for doc in retrieved_docs
            if doc.metadata.get("paper_id")
        }
        missing_ids = [
            paper_id for paper_id in selected_paper_ids
            if paper_id not in represented_ids
        ]
        if missing_ids:
            return (
                "⚠️ One or more selected papers could not provide "
                "retrievable evidence for this comparison.",
                []
            )
    else:
        # For explicit multi-PDF selection, _balanced_normal_retrieval has
        # already selected evidence per paper. Do NOT apply a second global
        # score gate here: that was the exact step that removed Request2.pdf



        # after it had been successfully retrieved.


        retrieved_docs = remove_redundant_documents(retrieved_docs)

        # Final relevance gate:
        # Do not show unrelated evidence just because vector search
        # found the nearest available chunks.
        if retrieved_docs:

            allowed_ids = list({
                d.metadata.get("paper_id")
                for d in retrieved_docs
                if d.metadata.get("paper_id")
            })

            gated_docs = _filter_relevant_documents(
                query,
                allowed_ids,
                k=len(retrieved_docs)
            )

            if not gated_docs:
                return (
                    "The uploaded research material does not contain "
                    "relevant information to answer this question.",
                    []
                )

            gated_keys = {
                (
                    d.metadata.get("paper_id"),
                    d.metadata.get("page"),
                    d.page_content
                )
                for d in gated_docs
            }

            retrieved_docs = [
                d for d in retrieved_docs
                if (
                    d.metadata.get("paper_id"),
                    d.metadata.get("page"),
                    d.page_content
                ) in gated_keys
            ]

            if not retrieved_docs:
                return (
                    "The uploaded research material does not contain "
                    "relevant information to answer this question.",
                    []
                )





    if comparison_mode:
        prompt_text = """
You are ResearchAI, an evidence-grounded research comparison assistant.
Use ONLY the RESEARCH CONTEXT below. Never infer methods, findings, audiences, titles, or conclusions from journal names, general knowledge, or typical practice.
If a fact is missing, write: "Not available in the retrieved excerpts."
Do not create a Markdown table. Use these exact sections:
## Source A
## Source B
(Use additional source sections only when present.)
## Similarities
## Differences
## Bottom Line
Every selected paper must be represented as a distinct source section.
Never compare two passages from the same paper as Source A and Source B.
Use the FILE and PAPER_ID labels in the context to identify each source.
Clearly attribute statements to the source represented in the context.
Do not invent a source that is not present.

RESEARCH CONTEXT:
{context}

USER QUESTION:
{question}
"""
    else:
        prompt_text = """
You are ResearchAI, an evidence-focused research assistant.
Answer using ONLY the RESEARCH CONTEXT below.
Never use outside knowledge, typical practices, assumptions, or inferred facts.
If the context does not support an answer, clearly say the information is not available in the uploaded research material.
Do not mention retrieval mechanics.

RESEARCH CONTEXT:
{context}

USER QUESTION:
{question}
"""

    prompt_template = ChatPromptTemplate.from_template(prompt_text)
    chat_model = ChatGroq(model="openai/gpt-oss-20b", api_key=st.session_state.get("groq_api_key"))
    response = (prompt_template | chat_model | StrOutputParser()).invoke({
        "context": format_docs(
            retrieved_docs,
            include_source_labels=comparison_mode
        ),
        "question": query
    })
    return response, retrieved_docs


def get_analysis_documents(selected_paper_ids=None):
    """Return balanced analysis passages grouped per allowed paper."""
    if selected_paper_ids:
        paper_ids = selected_paper_ids
    else:
        paper_ids = [p["paper_id"] for p in st.session_state.research_papers if p.get("processed", False)]
    if not paper_ids:
        return {}

    analysis_query = "research objective methodology methods experiment dataset results findings conclusion limitations discussion"
    grouped = {}
    for paper_id in paper_ids:
        docs = db.max_marginal_relevance_search(
            analysis_query, k=8, fetch_k=20, lambda_mult=0.5,
            filter={"paper_id": paper_id}
        )
        docs = remove_redundant_documents(docs)
        if docs:
            grouped[paper_id] = docs
    return grouped


def analyze_research_paper(document_docs):
    """Analyze one paper only. The caller is responsible for multi-paper separation."""
    if not document_docs:
        return None
    research_content = format_docs(document_docs)
    filename = document_docs[0].metadata.get("paper_filename", "Selected research paper")
    analysis_prompt = """
You are ResearchAI an evidence-focused research paper analyst.
Analyze ONLY the single paper represented below. Do not mix information from any other paper.
Never infer missing methods or findings. Do not use external knowledge.
Do not create Markdown tables.
Use exactly these headings, without bolding the # symbols:
## 🎯 Research Objective
## 🧪 Methodology
## 🔬 Key Findings
## ⚠️ Limitations
## 💡 Research Takeaways
For any section not supported by the excerpts, write exactly: Not clearly stated in the retrieved excerpts.

PAPER FILE: {filename}
RESEARCH EXCERPTS:
{research_content}
"""
    prompt = ChatPromptTemplate.from_template(analysis_prompt)
    model = ChatGroq(model="openai/gpt-oss-20b", api_key=st.session_state.get("groq_api_key"))
    return (prompt | model | StrOutputParser()).invoke({
        "filename": filename, "research_content": research_content
    })


def analyze_selected_library(selected_paper_ids=None):
    """Return explicit per-paper analyses; no cross-paper mixing."""
    grouped_docs = get_analysis_documents(selected_paper_ids)
    results = []
    for paper_id, docs in grouped_docs.items():
        filename = docs[0].metadata.get("paper_filename", "Research paper")
        analysis = analyze_research_paper(docs)
        if analysis:
            results.append({"paper_id": paper_id, "filename": filename, "analysis": analysis, "docs": docs})
    return results



def get_current_selected_paper_ids():
    """Read live paper checkbox values to avoid stale sidebar selection state."""
    selected_ids = []

    for paper in st.session_state.research_papers:
        widget_key = f"select_paper_{paper['paper_id']}"
        is_selected = st.session_state.get(
            widget_key,
            paper.get("selected", False)
        )

        if is_selected:
            selected_ids.append(paper["paper_id"])

    return selected_ids

def main():

    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

    if "research_papers" not in st.session_state:
        st.session_state.research_papers = []

    if "next_paper_id" not in st.session_state:
        st.session_state.next_paper_id = 1

    if "selected_paper_ids" not in st.session_state:
        st.session_state.selected_paper_ids = []

    # IMPORTANT:
    # Chroma persists on disk. Without this reset, an old run's vectors remain
    # and paper_id values (1, 2, 3...) can accidentally point to old PDFs.
    # Clear old vectors once at the beginning of each fresh app session.
    if "vector_db_session_initialized" not in st.session_state:
        clear_current_vector_database()
        st.session_state.vector_db_session_initialized = True


    st.set_page_config(
        page_title="ResearchAI",
        page_icon="🧠",
        layout="wide"
    )





     # Custom UI styling
    st.markdown(
        """
        <style>

        /* =========================
           GLOBAL APP BACKGROUND
        ========================= */

        .stApp {
            background:
                radial-gradient(
                    circle at 15% 10%,
                    rgba(99, 102, 241, 0.12),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 85% 20%,
                    rgba(190, 242, 100, 0.06),
                    transparent 25%
                ),
                #081120;
        }

        [data-testid="stHeader"] {
            background: rgba(8, 17, 32, 0.92);
        }

        .main .block-container {
            max-width: 1100px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }


        /* =========================
           TEXT
        ========================= */

        h1, h2, h3, h4 {
            color: #EAF0F8 !important;
            font-weight: 700 !important;
        }

        .stMarkdown,
        .stText,
        p,
        label {
            color: #B8C4D6 !important;
        }

        .stCaption {
            color: #7F8DA3 !important;
        }


        /* =========================
           TITLE
        ========================= */

        h1 {
            color: #F3F7FC !important;
            letter-spacing: -1px;
        }


        /* =========================
           DIVIDERS
        ========================= */

        hr {
            border: none !important;
            height: 1px !important;
            background: linear-gradient(
                90deg,
                transparent,
                #27364B,
                #9ACD32,
                #27364B,
                transparent
            ) !important;
            margin-top: 2rem !important;
            margin-bottom: 2rem !important;
        }


        /* =========================
           TEXT INPUT / QUESTION BOX
        ========================= */

        .stTextArea textarea,
        .stTextInput input {
            background-color: #0E1A2B !important;
            color: #EAF0F8 !important;

            border: 1px solid #26364C !important;

            border-radius: 14px !important;
        }

        .stTextArea textarea:focus,
        .stTextInput input:focus {
            border-color: #C6FF00 !important;
            box-shadow:
                0 0 0 1px #C6FF00,
                0 0 18px rgba(198, 255, 0, 0.12) !important;
        }

        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
            color: #65758B !important;
        }


        /* =========================
           FILE UPLOADER
        ========================= */

        [data-testid="stFileUploader"] {
            background: #0D1828 !important;
            border: 1px dashed #32445C !important;
            border-radius: 16px !important;
            padding: 8px !important;
        }

        [data-testid="stFileUploader"] button {
            background: #14243A !important;
            color: #C6FF00 !important;
            border: 1px solid #40536C !important;
            border-radius: 10px !important;
        }


        /* =========================
           DEFAULT BUTTONS
        ========================= */

        .stButton > button {
            background: linear-gradient(
                135deg,
                #16243A,
                #0F1A2B
            ) !important;

            color: #DCE5F0 !important;

            border: 1px solid #314158 !important;

            border-radius: 12px !important;

            font-weight: 700 !important;

            padding: 0.72rem 1rem !important;

            transition:
                all 0.25s ease !important;

            box-shadow:
                0 4px 14px rgba(0, 0, 0, 0.18) !important;
        }


        .stButton > button:hover {
            border-color: #C6FF00 !important;

            color: #C6FF00 !important;

            transform: translateY(-2px) !important;

            box-shadow:
                0 8px 22px rgba(198, 255, 0, 0.12) !important;
        }


        /* =========================
   PRIMARY GENERATION BUTTONS
========================= */

button[kind="primary"],
.stButton > button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #DFFF00,
        #A8E600
    ) !important;

    color: #050A12 !important;

    -webkit-text-fill-color: #050A12 !important;

    border: 1px solid #EEFF4D !important;

    font-weight: 900 !important;

    font-size: 16px !important;

    border-radius: 12px !important;

    text-shadow: none !important;

    opacity: 1 !important;

    box-shadow:
        0 6px 22px rgba(198, 255, 0, 0.22) !important;
}


/* Force ALL text inside primary button */
button[kind="primary"],
button[kind="primary"] *,
button[kind="primary"] p,
button[kind="primary"] span {
    color: #050A12 !important;
    opacity: 1 !important;
}


/* Hover */

button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(
        135deg,
        #EEFF4D,
        #C6FF00
    ) !important;

    color: #000000 !important;

    -webkit-text-fill-color: #000000 !important;

    transform: translateY(-2px) scale(1.01) !important;

    opacity: 1 !important;

    box-shadow:
        0 10px 28px rgba(198, 255, 0, 0.30) !important;
}

button[kind="primary"]:hover * {
    color: #050A12 !important;
}


        /* =========================
           SIDEBAR
        ========================= */

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #0B1627,
                    #07101D
                ) !important;

            border-right:
                1px solid #1D2B3D !important;
        }

        section[data-testid="stSidebar"] * {
            color: #D8E1EC !important;
        }


        /* =========================
           CHECKBOX
        ========================= */

        [data-testid="stCheckbox"] label {
            color: #B8C4D6 !important;
        }


        /* =========================
           EXPANDER
        ========================= */

        [data-testid="stExpander"] {
            background: #0D1828 !important;

            border: 1px solid #24354B !important;

            border-radius: 14px !important;
        }


        /* =========================
           ALERTS
        ========================= */

        [data-testid="stAlert"] {
            border-radius: 12px !important;
        }


        /* FORCE BUTTON TEXT VISIBILITY */

.stButton > button,
.stButton > button * {
    opacity: 1 !important;
}



.stButton > button[kind="primary"],
.stButton > button[kind="primary"] * {
    color: #050A12 !important;
    -webkit-text-fill-color: #050A12 !important;
    opacity: 1 !important;
    font-weight: 900 !important;
}
   


    /* =========================
   HIDE "PRESS ENTER TO APPLY"
========================= */

[data-testid="InputInstructions"] {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stTextInput"] [data-testid="InputInstructions"] {
    display: none !important;
    visibility: hidden !important;
}


        </style>
        """,
        unsafe_allow_html=True
    )








    # Main header
    st.title("🧠 ResearchAI")

    st.caption(
        "AI-powered research assistant for document-based insights"
    )

    st.divider()





    # Research Paper Analyzer
    st.markdown("## 📊 Research Paper Analyzer")

    st.caption(
        "Analyze selected papers separately, or analyze every processed paper when none is selected."
    )



    if st.button(
       "🔬 Analyze Research Library",
        use_container_width=True,
        type="primary"
):

        # Check whether at least one uploaded paper
        # has been successfully processed.
        has_processed_paper = any(
            paper.get("processed", False)
            for paper in st.session_state.research_papers
        )

        if not has_processed_paper:

            st.error(
                "❌ Invalid request: Please upload and "
                "process a research PDF before running analysis."
            )

        else:

            with st.spinner(
                "Analyzing research documents..."
            ):

                current_selected_paper_ids = get_current_selected_paper_ids()

                analysis_results = analyze_selected_library(
                    selected_paper_ids=current_selected_paper_ids
                )

                if not analysis_results:
                    st.warning("No research content was found for analysis.")
                else:
                    st.markdown("## 📑 Research Analysis")
                    if len(analysis_results) > 1:
                        st.caption("Each selected paper is analyzed separately to prevent cross-paper mixing.")
                    for item in analysis_results:
                        st.markdown(f"#### 📄 {item['filename']}")
                        st.markdown(item["analysis"])
                        with st.expander("📚 Analysis Evidence"):
                            for index, doc in enumerate(item["docs"], start=1):
                                source_name = doc.metadata.get("paper_filename", item["filename"])
                                page = doc.metadata.get("page")
                                page_display = f"Page {page + 1}" if page is not None else "Page unavailable"
                                st.markdown(f"**Source {index} • {source_name} • {page_display}**")
                                st.write(doc.page_content)





 # Query section

    st.markdown("<br>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("## 🔬 Research Assistant")






    st.caption(
        "Ask questions about your uploaded research documents."
    )

    query = st.text_area(
        "Research Question",
        placeholder="e.g., How is AI being used in modern drug discovery?",
        height=150
    )


    comparison_mode = st.checkbox(
        "📊 Compare research documents",
        help="Enable this when you want to compare findings across multiple research papers."
    )



    if st.button(
        "🧠 Analyze Research Question",
        use_container_width=True,
        type="primary"
):

        if not query:
            st.warning("Please enter a research question.")

        else:
            with st.spinner("Analyzing your research question..."):

                current_selected_paper_ids = get_current_selected_paper_ids()

                result, source_docs = run_rag_chain(
                    query=query,
                    comparison_mode=comparison_mode,
                    selected_paper_ids=current_selected_paper_ids
                )

            # Generated answer
            st.markdown("### 🧠 Research Insight")

            st.caption(
                "Generated from the most relevant passages "
                "in your research library."
            )

            st.write(result)

            # Supporting evidence
            st.divider()
            # Detect answers where the research library does not
# contain relevant information.
            no_evidence_phrases = [
                "does not contain relevant information",
                "does not contain information",
                "not available in the retrieved",
                "not available in the research material",
                "cannot be answered based on",
                "cannot be answered from",
                "no relevant information",
                "not supported by the provided research",
            ]

            answer_text_lower = str(result or "").lower()

            show_evidence = not any(
                phrase in answer_text_lower
                for phrase in no_evidence_phrases
            )

            st.markdown("### 📚 Research Evidence")

            if source_docs and show_evidence:

                st.caption(
                    f"{len(source_docs)} relevant research passages retrieved"
                )

                for index, doc in enumerate(
                    source_docs,
                    start=1
                ):

                    source_name = doc.metadata.get(
                        "paper_filename"
                    )

                    if not source_name:
                        source_path = doc.metadata.get(
                            "source",
                            "Unknown document"
                        )
                        source_name = os.path.basename(source_path)

                    page_number = doc.metadata.get(
                        "page"
                    )

                    if page_number is not None:
                        page_display = (
                            f"Page {page_number + 1}"
                        )
                    else:
                        page_display = "Page unavailable"

                    with st.expander(
                        f"📄 Source {index} • "
                        f"{source_name} • "
                        f"{page_display}"
                    ):

                        st.markdown(
                            "**Retrieved Evidence**"
                        )

                        st.write(
                            doc.page_content
                        )

            else:
                st.info(
                    "No supporting research evidence was retrieved."
                )




     # Sidebar
    with st.sidebar:

        st.markdown("## Research_AI")
        st.caption("Research Document Assistant")

        st.markdown("---")



        
        st.markdown("### 🔐 Groq API")

        groq_api_key = st.text_input(
            "Enter your Groq API key:",
             type="password",
            placeholder="Paste your Groq API key here...",
            key="groq_api_input"
    )

        if st.button(
            "Save & Verify API Key",
            use_container_width=True
  ):

           api_key = groq_api_key.strip()

           if not api_key:

               st.warning(
                  "⚠️ Please enter your Groq API key."
        )

           elif not api_key.startswith("gsk_"):

                st.error(
                "❌ Invalid Groq API key format. Groq API keys usually start with 'gsk_'."
        )

           else:
               try:
                   with st.spinner("🔄 Verifying your Groq API key..."):
                       test_model = ChatGroq(
                           model="openai/gpt-oss-20b",
                           api_key=api_key
                       )

                       test_model.invoke("Reply with only: OK")

                   # Save ONLY after successful verification
                   st.session_state.groq_api_key = api_key

                   st.success(
                       "✅ API key verified and saved successfully!"
                   )

               except Exception:
                   st.error(
                       "❌ Invalid or inactive Groq API key. Please check the key and try again."
                   )

        st.markdown("---")





                # Research Papers
        st.markdown("### 📚 Research Papers")

        st.caption(
            f"{len(st.session_state.research_papers)} / 5 papers added"
        )

        # Display uploaded papers
        for index, paper in enumerate(
            st.session_state.research_papers,
            start=1
        ):
            st.markdown(
                f"**📄 Paper {index}**"
            )

            st.caption(
                paper["filename"]
            )

            # Remove paper
            if st.button(
                "🗑 Remove",
                key=f"remove_paper_{paper['paper_id']}",
                use_container_width=True
            ):
                st.session_state.research_papers.pop(
                    index - 1
                )

                st.rerun()

            # Process this specific paper
            if not paper.get("processed", False):

                if st.button(
                    f"📥 Process Paper {index}",
                    key=f"process_paper_{paper['paper_id']}",
                    use_container_width=True
                ):
                    with st.spinner(
                        f"Processing Paper {index}..."
                    ):
                        add_to_db([paper])

                    paper["processed"] = True
                    st.session_state.documents_processed = True

                    st.success(
                        f"Paper {index} processed successfully."
                    )

                    st.rerun()

            else:

                st.success(
                    "✅ Processed"
                )

        # Paper Selection
        if st.session_state.research_papers:

            st.markdown("---")

            st.markdown(
                "### 🎯 Select Papers"
            )

            for paper in st.session_state.research_papers:

                paper["selected"] = st.checkbox(
                    paper["filename"],
                    value=paper.get(
                        "selected",
                        False
                    ),
                    key=f"select_paper_{paper['paper_id']}"
                )

            selected_papers = [
                paper
                for paper in st.session_state.research_papers
                if paper.get("selected", False)
            ]

            selected_paper_ids = [
                paper["paper_id"]
                for paper in selected_papers
            ]

            st.session_state.selected_paper_ids = selected_paper_ids




            st.caption(
                f"{len(selected_papers)} paper(s) selected"
            )

        # Add another paper
        if len(st.session_state.research_papers) < 5:

            st.markdown("")

            new_paper = st.file_uploader(
                "➕ Add another paper",
                type=["pdf"],
                key=(
                    f"paper_uploader_"
                    f"{st.session_state.next_paper_id}"
                )
            )

            if new_paper is not None:

                paper_exists = any(
                    paper["filename"] == new_paper.name
                    for paper in st.session_state.research_papers
                )

                if not paper_exists:

                    paper = {
                        "paper_id": (
                            st.session_state.next_paper_id
                        ),
                        "filename": new_paper.name,
                        "file": new_paper,
                        "processed": False,
                        "selected": False
                    }

                    st.session_state.research_papers.append(
                        paper
                    )

                    st.session_state.next_paper_id += 1

                    st.rerun()

                else:

                    st.warning(
                        "This paper has already been added."
                    )

        else:

            st.info(
                "Maximum 5 research papers reached."
            )





        



        st.markdown("---")

        st.caption(
            "Research_AI • Document-based RAG Assistant"
        )


if __name__ == "__main__":
    main()
