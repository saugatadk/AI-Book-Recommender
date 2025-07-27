# 📚 AI-Powered Book Recommender

A smart book recommendation tool powered by **NLP** and **semantic search**. Describe a story — in your own words — and this app finds books that match the meaning, tone, and genre you’re looking for.

It uses state-of-the-art **language embeddings**, **emotion tagging**, and **vector databases** to search over book descriptions with human-like understanding.

<!-- ![Book Recommender Demo](cover-example.png) -->

---

## 🚀 Features

- 🔍 **Semantic Search**: Describe a story, and it finds matching books using sentence-level embeddings.
- 🎭 **Emotion-Based Filtering**: Choose from tones like *Happy*, *Sad*, *Surprising*, and more.
- 📚 **Category-Based Browsing**: Filter by genres (Fiction, Nonfiction, etc.).
- 📸 **Beautiful Covers**: Clean image gallery with thumbnails and book details.
- ⚡ **Fast & Persistent**: Uses **Chroma** for fast vector search with on-disk persistence.

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [Chroma Vector DB](https://docs.trychroma.com/)
- [Gradio](https://www.gradio.app/)

---

## 🧠 How It Works

1. Loads a dataset of books with emotion scores (`joy`, `fear`, `sadness`, etc.)
2. Embeds book descriptions using `all-MiniLM-L6-v2`
3. Stores embeddings using ChromaDB
4. Accepts user description + optional filters (tone & category)
5. Ranks similar books using vector similarity + emotional tone

---

## 📦 Setup Instructions

1. **Clone the repo**:
   ```bash
   git clone https://github.com/saugatadk/AI-Book-Recommender.git
   cd AI-Book-Recommender
