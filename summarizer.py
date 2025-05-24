import yaml
import sqlite3
import requests
import time
import logging
import re
from typing import List, Dict, Any, Optional, Callable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def load_config():
    """Load configuration from application.yml file."""
    try:
        with open('application.yml', 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}  # Return empty config, defaults will be used


def get_paper_details(paper_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch paper details from the database."""
    config = load_config()
    db_path = config.get('database', {}).get('path', 'articles.db')

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        paper_details = []

        for paper_id in paper_ids:
            cursor.execute(
                "SELECT id, title, authors, abstract, pdf_url, published FROM papers WHERE id = ?",
                (paper_id,)
            )
            result = cursor.fetchone()
            if result:
                paper_details.append({
                    "id": result[0],
                    "title": result[1],
                    "authors": result[2],
                    "abstract": result[3],
                    "pdf_url": result[4],
                    "published": result[5] if result[5] else ""
                })

        conn.close()
        return paper_details
    except Exception as e:
        logger.error(f"Error getting paper details: {e}")
        return []


def create_apa_citation(paper: Dict[str, Any]) -> Dict[str, str]:
    """Create APA citation information for a paper."""
    try:
        authors = paper.get('authors', 'Unknown Author')
        if authors and authors != 'Unknown Author':
            first_author = authors.split(',')[0].strip()
            name_parts = first_author.split()
            if len(name_parts) > 1:
                last_name = name_parts[-1]
            else:
                last_name = first_author
        else:
            last_name = "Unknown"

        published = paper.get('published', '')
        if published:
            year_match = re.search(r'(\d{4})', published)
            year = year_match.group(1) if year_match else "n.d."
        else:
            year = "n.d."

        if authors.count(',') > 0:
            in_text = f"({last_name} et al., {year})"
        else:
            in_text = f"({last_name}, {year})"

        title = paper.get('title', 'Unknown Title')
        pdf_url = paper.get('pdf_url', '')

        if authors and authors != 'Unknown Author':
            formatted_authors = authors.replace(',', ', ')
        else:
            formatted_authors = "Unknown Author"

        full_reference = f"{formatted_authors} ({year}). {title}. arXiv preprint. {pdf_url}"

        return {
            "in_text": in_text,
            "full_reference": full_reference,
            "citation_key": f"{last_name.lower()}{year}"
        }
    except Exception as e:
        logger.error(f"Error creating APA citation: {e}")
        return {
            "in_text": "(Unknown, n.d.)",
            "full_reference": "Unknown Author (n.d.). Unknown Title.",
            "citation_key": "unknown"
        }


def fetch_paper_content(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Try to fetch the full content of a paper, fallback to abstract if needed."""
    try:
        if 'arxiv.org' in paper.get('pdf_url', ''):
            arxiv_id = paper['id']
            if not arxiv_id:
                arxiv_id = paper['pdf_url'].split('/')[-1].replace('.pdf', '')

            abstract_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            response = requests.get(abstract_url)

            if response.status_code == 200:
                return {
                    "title": paper['title'],
                    "authors": paper['authors'],
                    "content": paper['abstract'],
                    "source": "abstract",
                    "published": paper.get('published', ''),
                    "pdf_url": paper.get('pdf_url', ''),
                    "id": paper.get('id', '')
                }

        return {
            "title": paper['title'],
            "authors": paper['authors'],
            "content": paper['abstract'],
            "source": "abstract",
            "published": paper.get('published', ''),
            "pdf_url": paper.get('pdf_url', ''),
            "id": paper.get('id', '')
        }
    except Exception as e:
        logger.error(f"Error fetching paper content: {e}")
        return {
            "title": paper.get('title', 'Unknown Title'),
            "authors": paper.get('authors', 'Unknown Authors'),
            "content": paper.get('abstract', 'No abstract available'),
            "source": "error",
            "published": paper.get('published', ''),
            "pdf_url": paper.get('pdf_url', ''),
            "id": paper.get('id', '')
        }


def create_summary_prompt(papers_content: List[Dict[str, Any]], user_query: str) -> str:
    """Create a prompt for LLM to generate a summary with APA citations."""
    papers_text = ""
    citation_info = []

    for i, paper in enumerate(papers_content, 1):
        citation = create_apa_citation(paper)
        citation_info.append(citation)

        papers_text += f"\n\nPAPER {i} [Use this citation when referencing: {citation['in_text']}]:\n"
        papers_text += f"Title: {paper['title']}\n"
        papers_text += f"Authors: {paper['authors']}\n"
        papers_text += f"Content: {paper['content']}\n"

    prompt = f"""You are an academic research assistant. You need to create a comprehensive summary article based on the following research papers related to this query: "{user_query}".

IMPORTANT CITATION INSTRUCTIONS:
- You MUST include in-text citations in APA format when referencing each paper
- Use the provided citation format for each paper exactly as shown
- When discussing findings from a specific paper, always include the in-text citation immediately after the claim
- You can reference multiple papers in one sentence like: (Smith, 2023; Johnson, 2024)
- Make sure every major claim or finding is properly cited
- DO NOT create a references section - this will be handled separately

{papers_text}

Create a well-structured research summary article that:
1. Begins with an introduction that establishes the context and importance of this research area
2. Synthesizes the main findings, methodologies, and contributions from all papers WITH PROPER IN-TEXT CITATIONS
3. Identifies common themes, contradictions, and research gaps WITH CITATIONS
4. Discusses practical implications of the research WITH CITATIONS
5. Concludes with future research directions

CITATION REQUIREMENTS:
- Every time you mention a finding, method, or idea from a paper, include the in-text citation
- Use the exact citation format provided for each paper
- Example: "Recent studies have shown significant improvements in accuracy (Smith, 2023) while other research indicates..." 

The summary should be scholarly in tone, well-organized with clear sections, and approximately 1000-1500 words.
Include a title for the summary article.
Format the output as markdown with appropriate headers, lists, and emphasis.

"""

    return prompt


def collect_papers_content(paper_ids: List[str], progress_callback: Optional[ProgressCallback] = None) -> List[
    Dict[str, Any]]:
    """Collect content for all papers."""
    if progress_callback:
        progress_callback(0.05, "Fetching paper details...")

    paper_details = get_paper_details(paper_ids)
    if not paper_details:
        return []

    if progress_callback:
        progress_callback(0.1, "Fetching paper content...")

    papers_content = []
    for i, paper in enumerate(paper_details):
        if progress_callback:
            progress = 0.1 + (0.4 * (i / len(paper_details)))
            progress_callback(progress, f"Processing paper {i + 1} of {len(paper_details)}...")

        content = fetch_paper_content(paper)
        papers_content.append(content)
        time.sleep(1)

    return papers_content


def summarize_papers(paper_ids: List[str], user_query: str, progress_callback: Optional[ProgressCallback] = None,
                     model_choice: str = "openai"):
    """Main function to summarize papers with the specified model."""
    if model_choice == "openai":
        from summarizer_api import generate_summary_openai

        papers_content = collect_papers_content(paper_ids, progress_callback)
        if not papers_content:
            return "No papers found with the provided IDs."

        if progress_callback:
            progress_callback(0.5, "Generating summary with OpenAI...")

        return generate_summary_openai(papers_content, user_query, progress_callback)

    elif model_choice == "ollama":
        from summarizer_ollama import generate_summary_ollama

        papers_content = collect_papers_content(paper_ids, progress_callback)
        if not papers_content:
            return "No papers found with the provided IDs."

        if progress_callback:
            progress_callback(0.5, "Generating summary with Ollama...")

        return generate_summary_ollama(papers_content, user_query, progress_callback)

    else:
        raise ValueError("Unsupported model choice. Use 'openai' or 'ollama'.")