# Project Modules and Technologies Report (Draft)

## 1. Project Overview
This project is a comprehensive Football Analytics Platform designed to identify undervalued players, compare player statistics, find similar players, and provide an interactive chatbot for real-time queries. The application is built using **Flask** as the backend framework and utilizes advanced AI and Machine Learning techniques for data analysis.

## 2. Modules and Functionalities

### 2.1 Undervalued Players Module
**Description:**
This module identifies players whose market value is lower than their predicted value based on performance metrics. It allows users to filter players by position, league, squad, age, and market value to find potential "Moneyball" targets.

**Tools & Algorithms:**
*   **Data Processing:** `pandas` is used for loading, filtering, and sorting the dataset (`all_predictions_with_undervaluation.csv`).
*   **Filtering Logic:** Custom filtering algorithms based on user inputs (min/max age, market value, undervaluation score).
*   **Backend:** Flask routes (`/api/undervalued`) handle the logic and return JSON data for the frontend.

### 2.2 Similar Players Module
**Description:**
This module finds players who are statistically similar to a selected player. It helps scouts and analysts find replacements or alternatives with comparable playing styles.

**Tools & Algorithms:**
*   **Similarity Algorithm:** **Cosine Similarity** (`sklearn.metrics.pairwise.cosine_similarity`) is used to calculate the distance between player feature vectors.
*   **Feature Engineering:** Players are grouped by position (Attacker, Midfielder, Defender, Goalkeeper). Features are normalized using **Min-Max Scaling** (`sklearn.preprocessing.MinMaxScaler`) before comparison.
*   **Library:** `scikit-learn` for similarity and preprocessing.

### 2.3 Player Comparison Module
**Description:**
This module allows users to compare two or more players side-by-side. It visualizes their stats using radar charts and generates an AI-powered textual comparison report highlighting strengths, weaknesses, and tactical suitability.

**Tools & Algorithms:**
*   **AI Generation:** **Google Gemini API** (`gemini-2.0-flash`) is used to generate the qualitative comparison report.
*   **Visualization:** Radar charts (Frontend JS) powered by backend data preparation.
*   **Prompt Engineering:** Custom prompts are designed to instruct the AI to act as an elite European football scouting analyst.

### 2.4 AI Chatbot Module (RAG & Live Data)
**Description:**
An intelligent chatbot that answers user queries about players, teams, and matches. It combines static database knowledge with live real-time data.

**Tools & Algorithms:**
*   **RAG (Retrieval-Augmented Generation):**
    *   **Framework:** `LangChain` is used to orchestrate the RAG pipeline.
    *   **Vector Database:** **FAISS** (Facebook AI Similarity Search) stores player data embeddings for fast retrieval.
    *   **Embeddings:** `HuggingFaceEmbeddings` (model: `all-MiniLM-L6-v2`) converts text data into vector representations.
    *   **LLM:** **Google Gemini** (`gemini-1.5-flash`) generates natural language responses based on retrieved context.
*   **Live Data Integration:**
    *   **API-Football (api-sports.io):** Fetches real-time match scores, live fixtures, and recent team results.
    *   **SofaScore RapidAPI:** Retrieves TV broadcast information for matches.

### 2.5 Data Processing & Ingestion
**Description:**
Scripts responsible for cleaning data and preparing the vector database for the chatbot.

**Tools & Algorithms:**
*   **Vectorization:** `create_vector_db.py` uses `CSVLoader` to read data and `FAISS` to index it.
*   **Data Cleaning:** `pandas` and `numpy` are used for handling missing values and data type conversions.

## 3. Technology Stack Summary

| Category | Technology / Tool | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | **Flask** | Web server and API endpoints. |
| **Data Manipulation** | **Pandas, NumPy** | Data cleaning, filtering, and analysis. |
| **Machine Learning** | **Scikit-learn** | Cosine similarity, normalization (MinMax). |
| **AI / LLM** | **Google Gemini API** | Generative AI for reports and chatbot responses. |
| **RAG Framework** | **LangChain** | Orchestrating retrieval and generation. |
| **Vector Database** | **FAISS** | Storing and searching data embeddings. |
| **Embeddings** | **HuggingFace (all-MiniLM-L6-v2)** | Creating vector representations of text. |
| **External APIs** | **API-Football, SofaScore** | Real-time football data and TV listings. |
