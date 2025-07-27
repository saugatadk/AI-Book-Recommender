import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr
import os

load_dotenv()

# Load books data
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

#Intialize embeddings model
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH = "chroma_store"


# Optimized database initialization
def initialize_vector_db():
    """Initialize vector database efficiently"""
    if os.path.exists(CHROMA_PATH):
        print("Loading existing vector database...")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=huggingface_embeddings)
    else:
        print("Creating new vector database...")
        # Only load and process documents if database doesn't exist
        raw_documents = TextLoader("tagged_descriptions.txt").load()
        documents = []
        for doc in raw_documents:
            lines = doc.page_content.split("\n")
            for line in lines:
                line = line.strip()
                if line:
                    documents.append(Document(page_content=line))

        db = Chroma.from_documents(
            documents,
            embedding=huggingface_embeddings,
            persist_directory=CHROMA_PATH
        )
        print(f"Vector database saved to {CHROMA_PATH}")
        return db

# Initialize the database
db_books = initialize_vector_db()

# raw_documents = TextLoader("tagged_descriptions.txt").load()
# # Create a document per line
# documents = []
# for doc in raw_documents:
#     lines = doc.page_content.split("\n")
#     for line in lines:
#         line = line.strip()
#         if line:
#             documents.append(Document(page_content=line))
#
# huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db_books = Chroma.from_documents(documents, embedding=huggingface_embeddings)
#
#
# huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# CHROMA_PATH = "chroma_store"
#
# if os.path.exists(CHROMA_PATH):
#     db_books = Chroma(persist_directory=CHROMA_PATH, embedding_function=huggingface_embeddings)
# else:
#     db_books = Chroma.from_documents(
#         documents,
#         embedding=huggingface_embeddings,
#         persist_directory=CHROMA_PATH
#     )

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search_with_score(query, k = initial_top_k)

    #Parse results and extract ISBN + score
    recommendations = []
    seen_isbns = set()
    for doc, score in recs:
        # Assume the ISBN is the first token (or improve based on your format)
        isbn_str = doc.page_content.strip('"').split()[0]
        try:
            isbn = int(isbn_str)
        except ValueError:
            continue  # skip bad lines

        if isbn not in seen_isbns:
            recommendations.append((isbn, score))
            seen_isbns.add(isbn)

    #Build DataFrame from matching ISBNs and scores
    rec_df = pd.DataFrame(recommendations, columns=["isbn13", "similarity_score"])

    #Merge with book metadata
    final_df = rec_df.merge(books, on="isbn13", how="inner")

    #Sort by similarity score
    book_recs = final_df.sort_values(by="similarity_score", ascending=True).head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(
        query: str,
        category: str = None,
        tone: str = None,
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:20]) + "..."

        authors = row.get("authors", "")
        if pd.isna(authors) or not isinstance(authors, str):
            authors_str = "Unknown Author"
        else:
            authors_split = authors.split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = authors_split[0]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# AI Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about love")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a Category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select a emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations", variant = "primary")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)

    if __name__ == "__main__":
        dashboard.launch()