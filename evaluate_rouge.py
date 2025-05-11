import os
import sqlite3
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import re
import io
import base64


def load_config():
    """Load configuration from YAML file."""
    try:
        with open('application.yml', 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}


def get_paper_details(paper_ids):
    """Fetch paper details from the database."""
    config = load_config()
    db_path = config.get('database', {}).get('path', 'articles.db')

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        paper_details = []
        for paper_id in paper_ids:
            cursor.execute(
                "SELECT id, title, authors, abstract, pdf_url FROM papers WHERE id = ?",
                (paper_id,)
            )
            result = cursor.fetchone()
            if result:
                paper_details.append({
                    "id": result[0],
                    "title": result[1],
                    "authors": result[2],
                    "abstract": result[3],
                    "pdf_url": result[4]
                })

        conn.close()
        return paper_details
    except Exception as e:
        print(f"Error getting paper details: {e}")
        return []


def create_reference_text(papers):
    """Create reference text from papers for ROUGE evaluation."""
    reference_text = ""
    for i, paper in enumerate(papers, 1):
        reference_text += f"PAPER {i}\n"
        reference_text += f"Title: {paper['title']}\n"
        reference_text += f"Authors: {paper['authors']}\n"
        reference_text += f"Abstract: {paper['abstract']}\n\n"
    return reference_text


def compute_rouge(reference, candidate):
    """Compute ROUGE scores between reference and candidate text."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores


def rouge_to_dataframe(rouge_openai, rouge_ollama):
    """Convert ROUGE scores to a DataFrame for display and visualization."""
    df = pd.DataFrame({
        "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "OpenAI_Precision": [
            rouge_openai['rouge1'].precision,
            rouge_openai['rouge2'].precision,
            rouge_openai['rougeL'].precision
        ],
        "OpenAI_Recall": [
            rouge_openai['rouge1'].recall,
            rouge_openai['rouge2'].recall,
            rouge_openai['rougeL'].recall
        ],
        "OpenAI_F1": [
            rouge_openai['rouge1'].fmeasure,
            rouge_openai['rouge2'].fmeasure,
            rouge_openai['rougeL'].fmeasure
        ],
        "Ollama_Precision": [
            rouge_ollama['rouge1'].precision,
            rouge_ollama['rouge2'].precision,
            rouge_ollama['rougeL'].precision
        ],
        "Ollama_Recall": [
            rouge_ollama['rouge1'].recall,
            rouge_ollama['rouge2'].recall,
            rouge_ollama['rougeL'].recall
        ],
        "Ollama_F1": [
            rouge_ollama['rouge1'].fmeasure,
            rouge_ollama['rouge2'].fmeasure,
            rouge_ollama['rougeL'].fmeasure
        ]
    })
    return df


def generate_comparison_chart(df):
    """Generate a comparison chart of ROUGE scores."""
    metrics = df["Metric"]
    x = range(len(metrics))

    plt.figure(figsize=(10, 6))

    # F1 Scores
    plt.subplot(1, 3, 1)
    plt.bar([i - 0.2 for i in x], df["OpenAI_F1"], width=0.4, label='OpenAI', color='#4285F4')
    plt.bar([i + 0.2 for i in x], df["Ollama_F1"], width=0.4, label='Ollama', color='#34A853')
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("F1 Scores")
    plt.legend()

    # Precision
    plt.subplot(1, 3, 2)
    plt.bar([i - 0.2 for i in x], df["OpenAI_Precision"], width=0.4, label='OpenAI', color='#4285F4')
    plt.bar([i + 0.2 for i in x], df["Ollama_Precision"], width=0.4, label='Ollama', color='#34A853')
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Precision")

    # Recall
    plt.subplot(1, 3, 3)
    plt.bar([i - 0.2 for i in x], df["OpenAI_Recall"], width=0.4, label='OpenAI', color='#4285F4')
    plt.bar([i + 0.2 for i in x], df["Ollama_Recall"], width=0.4, label='Ollama', color='#34A853')
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Recall")

    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf


def generate_radar_chart(df):
    """Generate a radar chart comparing OpenAI and Ollama metrics."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # Setup radar chart
    categories = ['ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1',
                  'ROUGE-1 Precision', 'ROUGE-2 Precision', 'ROUGE-L Precision',
                  'ROUGE-1 Recall', 'ROUGE-2 Recall', 'ROUGE-L Recall']

    values_openai = [
        df.loc[0, 'OpenAI_F1'], df.loc[1, 'OpenAI_F1'], df.loc[2, 'OpenAI_F1'],
        df.loc[0, 'OpenAI_Precision'], df.loc[1, 'OpenAI_Precision'], df.loc[2, 'OpenAI_Precision'],
        df.loc[0, 'OpenAI_Recall'], df.loc[1, 'OpenAI_Recall'], df.loc[2, 'OpenAI_Recall']
    ]

    values_ollama = [
        df.loc[0, 'Ollama_F1'], df.loc[1, 'Ollama_F1'], df.loc[2, 'Ollama_F1'],
        df.loc[0, 'Ollama_Precision'], df.loc[1, 'Ollama_Precision'], df.loc[2, 'Ollama_Precision'],
        df.loc[0, 'Ollama_Recall'], df.loc[1, 'Ollama_Recall'], df.loc[2, 'Ollama_Recall']
    ]

    # Number of variables
    N = len(categories)

    # Angles for each category
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Add values for both models
    values_openai += values_openai[:1]
    values_ollama += values_ollama[:1]

    # Plot
    ax.plot(angles, values_openai, linewidth=2, linestyle='solid', label="OpenAI", color='#4285F4')
    ax.fill(angles, values_openai, alpha=0.25, color='#4285F4')

    ax.plot(angles, values_ollama, linewidth=2, linestyle='solid', label="Ollama", color='#34A853')
    ax.fill(angles, values_ollama, alpha=0.25, color='#34A853')

    # Set category labels
    plt.xticks(angles[:-1], categories)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title('OpenAI vs Ollama ROUGE Metrics Comparison', size=15, color='gray', y=1.1)

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf


def evaluate_summaries(paper_ids, user_query, summary_api, summary_ollama):
    """Evaluate summaries using ROUGE scores and generate visualizations."""
    # Get paper details
    papers = get_paper_details(paper_ids)
    if not papers:
        return None, None, None

    # Create reference text
    reference_text = create_reference_text(papers)

    # Compute ROUGE scores
    rouge_openai = compute_rouge(reference_text, summary_api)
    rouge_ollama = compute_rouge(reference_text, summary_ollama)

    # Convert to DataFrame
    df = rouge_to_dataframe(rouge_openai, rouge_ollama)

    # Generate charts
    comparison_chart = generate_comparison_chart(df)
    radar_chart = generate_radar_chart(df)

    return df, comparison_chart, radar_chart


def slugify(text):
    """Convert text to slug format for filenames."""
    text = text.lower()
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    return text


def save_evaluation_results(user_query, reference_text, summary_api, summary_ollama, df, comparison_chart,
                            radar_chart):
    """Save evaluation results to disk."""
    # Create folder structure
    query_slug = slugify(user_query)
    base_dir = "evaluation_results"
    output_dir = os.path.join(base_dir, query_slug)
    os.makedirs(output_dir, exist_ok=True)

    # Save text files
    with open(os.path.join(output_dir, 'reference.txt'), 'w') as f:
        f.write(reference_text)
    with open(os.path.join(output_dir, 'summary_api.md'), 'w') as f:
        f.write(summary_api)
    with open(os.path.join(output_dir, 'summary_ollama.md'), 'w') as f:
        f.write(summary_ollama)

    # Save metrics
    df.to_csv(os.path.join(output_dir, "rouge_metrics.csv"), index=False)

    # Save charts
    comparison_chart.seek(0)
    with open(os.path.join(output_dir, "rouge_comparison.png"), 'wb') as f:
        f.write(comparison_chart.getvalue())

    radar_chart.seek(0)
    with open(os.path.join(output_dir, "radar_comparison.png"), 'wb') as f:
        f.write(radar_chart.getvalue())

    return output_dir