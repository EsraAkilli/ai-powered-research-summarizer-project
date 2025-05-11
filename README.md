# AI-Powered Research Summarizer

AI-Powered Research Summarizer is a system that retrieves academic papers from ArXiv, processes them using TF-IDF & LSA techniques, and generates a comprehensive summary using the OpenAI API.

## Features

- **ArXiv API Integration** – Fetches relevant academic papers based on user-defined topics.
- **TF-IDF & LSA Processing** – Vectorizes and analyzes the papers for similarity.
- **Cosine Similarity Scoring** – Ranks papers based on their relevance to the user’s research query.
- **OpenAI API Summarization** – Generates a structured summary (introduction, main points, and conclusion) from the selected papers.
- **Streamlit UI** – Provides an interactive interface for users.

## Installation

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/ai-powered-research-summarizer.git
cd ai-powered-research-summarizer
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
pip install pandas requests streamlit xmltodict scikit-learn nltk openai
```

4. Configure OpenAI API Key
Create a application.yml file and add the following:
```sh
OPENAI_API_KEY: "your-openai-api-key"
```

5. Run the Application
```sh
streamlit run app.py
```

## Usage
1. Select a Research Topic: Choose a subject and enter a keyword.
2. Fetch Papers: Retrieve relevant academic papers from ArXiv and store them.
3. Find Similar Papers: Analyze papers using TF-IDF & LSA and rank them based on similarity.
4. Generate a Summary: Use OpenAI API to generate a structured summary.
5. View Results: The system displays the summarized research along with the most relevant papers.