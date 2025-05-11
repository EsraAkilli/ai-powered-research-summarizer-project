# AI-Powered Research Summarizer

AI-Powered Research Summarizer is an interactive research assistant that fetches academic papers from ArXiv, analyzes their relevance using NLP techniques (TF-IDF & LSA), generates summaries via OpenAI or Ollama LLMs, and evaluates the results with ROUGE scores and professional APA-style references.

## Features

- **ArXiv API Integration** – Fetches relevant academic papers based on user-defined topics.
- **TF-IDF & LSA Processing** – Vectorizes and analyzes the papers for similarity.
- **Cosine Similarity Scoring** – Ranks papers based on their relevance to the user’s research query.
- **LLM-Based Summarization** – Generates a structured summary (introduction, main points, and conclusion) from the selected papers.
- **Streamlit UI** – Provides an interactive interface for users.
- **ROUGE Evaluation** – Compares the summaries from both models using ROUGE-1, ROUGE-2, ROUGE-L metrics.

## Installation

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/ai-powered-research-summarizer-project.git
cd ai-powered-research-summarizer-project
```

### 2. Create and Activate a Virtual Environment
MacOS/Linux
```sh
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies
```sh
pip install -r requirements.txt
```

If requirements.txt is missing or you encounter issues, install dependencies manually:
```sh
pip install pandas requests streamlit openai pyyaml scikit-learn nltk matplotlib rouge-score
```

4. Configure OpenAI API Key
Create a application.yml file and add the following:
```sh
OPENAI_API_KEY: "your-openai-api-key"
```

5. Run the Application
```sh
streamlit run main.py
```

## Usage
1. Select a Research Topic: Choose a subject and enter a keyword.
2. Fetch Papers: Retrieve relevant academic papers from ArXiv and store them.
3. Find Similar Papers: Analyze papers using TF-IDF & LSA and rank them based on similarity.
4. Generate a Summary: Choose between OpenAI or Ollama to generate a structured summary.
5. View Results: The system displays the summarized research along with the most relevant papers.