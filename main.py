import pandas as pd
import requests
import sqlite3
import streamlit as st
from datetime import datetime, date
import xml.etree.ElementTree as ET
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import summarizer
import evaluate_rouge

nltk.download('punk')
nltk.download('stopwords')

ARXIV_SUBJECT_MAP = {
    "Physics": "physics.*",
    "Mathematics": "math.*",
    "Quantitative Biology": "q-bio.*",
    "Computer Science": "cs.*",
    "Quantitative Finance": "q-fin.*",
    "Statistics": "stat.*",
    "Electrical Engineering and Systems Science": "eess.*",
    "Economics": "econ.*"
}

def create_database():
    conn = sqlite3.connect('articles.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            topic TEXT,
            categories TEXT,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            published DATE,
            updated DATE,
            doi TEXT,
            pdf_url TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT UNIQUE,
            topic TEXT
        )
    ''')
    conn.commit()
    conn.close()

STOP_WORDS = set(stopwords.words('english'))

def text_preprocessing(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    processed_text = " ".join(filtered_tokens)
    return processed_text

def compute_similarity(papers, user_query):
    if not papers:
        st.error("No papers found for this query!")
        return [], None, None, None
    abstracts = [text_preprocessing(paper['abstract']) for paper in papers]
    if len(abstracts) < 2:
        st.error("Not enough data to compute similarity.")
        return [], None, None, None
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)
    n_components = max(1, min(len(abstracts), 100))
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    query_vector = tfidf_vectorizer.transform([text_preprocessing(user_query)])
    query_reduced = svd.transform(query_vector)
    similarities = cosine_similarity(query_reduced, reduced_matrix).flatten()
    top_indices = similarities.argsort()[-5:][::-1]
    return [(papers[i], similarities[i]) for i in top_indices], tfidf_matrix, reduced_matrix, tfidf_vectorizer

def fetch_arxiv_papers(arxiv_category, start_date, end_date, max_results=300, sort_by="relevance", batch_size=50):
    base_url = "https://export.arxiv.org/api/query?"
    start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d") + "0000"
    end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d") + "2359"
    all_papers = []
    start_index = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    while start_index < max_results:
        current_batch_size = min(batch_size, max_results - start_index)
        status_text.text(f"Fetching papers {start_index + 1}-{start_index + current_batch_size} of {max_results}...")
        query_params = {
            "search_query": f"cat:{arxiv_category} AND submittedDate:[{start} TO {end}]",
            "start": start_index,
            "max_results": current_batch_size,
            "sortBy": sort_by
        }
        response = requests.get(base_url, params=query_params)
        if response.status_code == 200:
            batch_papers = parse_response(response.text)
            all_papers.extend(batch_papers)
            progress = min(1.0, len(all_papers) / max_results)
            progress_bar.progress(progress)
            if len(batch_papers) < current_batch_size:
                break
            start_index += current_batch_size
            time.sleep(3)
        else:
            status_text.text(f"API request failed: {response.status_code}")
            return []
    progress_bar.progress(1.0)
    status_text.text(f"Successfully fetched {len(all_papers)} papers.")
    return all_papers

def parse_response(xml_data):
    root = ET.fromstring(xml_data)
    papers = []
    entries = root.findall('{http://www.w3.org/2005/Atom}entry')
    for entry in entries:
        if entry.find('{http://www.w3.org/2005/Atom}id') is None:
            continue
        category_elem = entry.find('{http://www.w3.org/2005/Atom}category')
        category_term = category_elem.attrib['term'] if category_elem is not None else ''
        pdf_links = [
            link.attrib['href']
            for link in entry.findall('{http://www.w3.org/2005/Atom}link')
            if 'title' in link.attrib and link.attrib['title'] == 'pdf'
        ]
        pdf_url = pdf_links[0] if pdf_links else ''
        paper = {
            'id': entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1],
            'title': (entry.find('{http://www.w3.org/2005/Atom}title').text or '').strip(),
            'authors': ', '.join(
                author.find('{http://www.w3.org/2005/Atom}name').text
                for author in entry.findall('{http://www.w3.org/2005/Atom}author')
            ),
            'abstract': (entry.find('{http://www.w3.org/2005/Atom}summary').text or '').strip(),
            'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
            'updated': entry.find('{http://www.w3.org/2005/Atom}updated').text,
            'categories': category_term,
            'doi': '',
            'pdf_url': pdf_url
        }
        papers.append(paper)
    return papers

def save_to_database(papers, subject_label):
    conn = sqlite3.connect('articles.db')
    c = conn.cursor()
    for paper in papers:
        c.execute("SELECT id FROM papers WHERE id = ?", (paper['id'],))
        existing_paper = c.fetchone()
        if not existing_paper:
            c.execute('''
                INSERT INTO papers (id, topic, title, categories, authors, abstract, published, updated, doi, pdf_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper['id'],
                subject_label,
                paper['title'],
                paper['categories'],
                paper['authors'],
                paper['abstract'],
                paper['published'],
                paper['updated'],
                paper['doi'],
                paper['pdf_url']
            ))
    conn.commit()
    conn.close()

def save_query(user_query, subject_label):
    conn = sqlite3.connect('articles.db')
    c = conn.cursor()
    c.execute('INSERT OR IGNORE INTO queries (query_text, topic) VALUES (?, ?)', (user_query, subject_label))
    conn.commit()
    conn.close()

def get_previous_queries(subject_label):
    conn = sqlite3.connect('articles.db')
    c = conn.cursor()
    c.execute('SELECT query_text FROM queries WHERE topic = ?', (subject_label,))
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results

def main():
    st.title("AI-Powered Summarization System")
    create_database()

    with st.expander("About the App"):
        st.write("This system fetches academic papers and prepares summaries using NLP techniques.")

    subject_options = list(ARXIV_SUBJECT_MAP.keys())
    selected_subject = st.selectbox("Select Research Topic", subject_options, key="global_subject")

    tab1, tab2, tab3 = st.tabs(["Fetch Papers", "Literature Review", "Evaluate Summaries"])
    max_allowed_date = date(2025, 12, 31)

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            start_dt = st.date_input("Start Date", value=date(2023, 1, 1), min_value=date(2000, 1, 1), max_value=max_allowed_date)
        with col2:
            end_dt = st.date_input("End Date", value=min(date.today(), max_allowed_date), min_value=date(2000, 1, 1), max_value=max_allowed_date)

        sort_by = st.selectbox("Sort By", ["relevance", "lastUpdatedDate", "submittedDate"], index=0)

        max_results = st.slider("Maximum Papers to Retrieve", min_value=10, max_value=300, value=100)

        if st.button("Fetch and Save Papers"):
            arxiv_category = ARXIV_SUBJECT_MAP[selected_subject]
            papers = fetch_arxiv_papers(arxiv_category, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), max_results, sort_by)
            save_to_database(papers, selected_subject)
            st.success("✅ Papers successfully fetched.")

    with tab2:
        previous_queries = get_previous_queries(selected_subject)
        st.subheader("Select from previous queries or type a new one")
        col1, col2 = st.columns(2)

        with col1:
            selected_query = st.selectbox("Previous Queries", options=[""] + previous_queries)
        with col2:
            user_query = st.text_input("Or type a new query", value="", key="query_review")

        final_query = user_query if user_query else selected_query

        if st.button("Find Similar Papers", key="find_similar_tab2"):
            if not final_query:
                st.error("Please enter or select a research interest.")
            else:
                st.session_state['user_query'] = final_query
                save_query(final_query, selected_subject)
                conn = sqlite3.connect('articles.db')
                c = conn.cursor()
                c.execute("SELECT id, title, authors, abstract, published, updated, categories, pdf_url FROM papers WHERE topic = ?", (selected_subject,))
                rows = c.fetchall()
                conn.close()
                papers = [{ "id": row[0], "title": row[1], "authors": row[2], "abstract": row[3], "published": row[4], "updated": row[5], "categories": row[6], "pdf_url": row[7] } for row in rows]
                similar_papers, tfidf_matrix, reduced_matrix, tfidf_vectorizer = compute_similarity(papers, final_query)
                if similar_papers:
                    st.session_state['similar_papers'] = similar_papers
                    st.session_state['tfidf_matrix'] = tfidf_matrix
                    st.session_state['reduced_matrix'] = reduced_matrix
                    st.session_state['vectorizer'] = tfidf_vectorizer
                    st.success("✅ Similar papers identified successfully!")

                with st.expander("Show Technical Analysis Details"):
                    st.subheader("🔹 Preprocessed Text Samples")
                    abstracts = [text_preprocessing(paper['abstract']) for paper, _ in st.session_state['similar_papers']]
                    for i, text in enumerate(abstracts):
                        st.write(f"**Abstract {i + 1}:** {text[:300]}...")

                    st.subheader("📊 TF-IDF Vectorization")
                    st.write("**TF-IDF Shape:**", st.session_state['tfidf_matrix'].shape)
                    st.dataframe(pd.DataFrame(st.session_state['tfidf_matrix'].toarray()).iloc[:5, :10])

                    st.subheader("🔻 LSA (Latent Semantic Analysis) Applied")
                    st.write("**Reduced Matrix Shape:**", st.session_state['reduced_matrix'].shape)
                    st.dataframe(pd.DataFrame(st.session_state['reduced_matrix']).head(5))

                    st.subheader("📈 Cosine Similarity Scores")
                    similarity_df = pd.DataFrame(
                        {"Paper": [f"Paper {i + 1}" for i in range(len(st.session_state['similar_papers']))],
                         "Similarity": [score for _, score in st.session_state['similar_papers']]}
                    )
                    similarity_df = similarity_df.sort_values(by="Similarity", ascending=False)
                    st.dataframe(similarity_df.head(10))
                    st.write("**Similar Papers Shape:**", similarity_df.shape)

        if 'similar_papers' in st.session_state and st.session_state['similar_papers']:
            st.subheader("Top 5 Most Relevant Papers")
            for i, (paper, score) in enumerate(st.session_state['similar_papers']):
                with st.expander(f"{i + 1}. {paper['title']} (Similarity Score: {score:.2f})"):
                    st.write(f"**Authors:** {paper['authors']}")
                    st.write(f"**Published:** {paper['published']}")
                    st.write(f"**Updated:** {paper['updated']}")
                    st.write(f"**Categories:** {paper['categories']}")
                    st.write("**Abstract:**")
                    st.write(paper['abstract'])
                    if paper['pdf_url']:
                        st.markdown(f"[Download PDF]({paper['pdf_url']})")


            st.subheader("Generate Research Summary")
            st.write("Select one of the following methods to configure the summarization system.")
            model_choice = st.selectbox("Summarization methods", ["openai", "ollama"], index=0)

            if st.button("Generate Summary", key="generate_summary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                summary_container = st.container()

                def update_progress(progress, status_text_msg):
                    progress_bar.progress(progress)
                    status_text.text(status_text_msg)

                paper_ids = [paper['id'] for paper, _ in st.session_state['similar_papers']]
                summary = summarizer.summarize_papers(
                    paper_ids,
                    st.session_state['user_query'],
                    update_progress,
                    model_choice=model_choice
                )

                if model_choice == "openai":
                    st.session_state['summary_api'] = summary
                elif model_choice == "ollama":
                    st.session_state['summary_ollama'] = summary

                references = []
                for paper, _ in st.session_state['similar_papers']:
                    authors = paper['authors'].replace(',', ', ') if paper['authors'] else "Unknown Author"
                    year = paper['published'][:4] if paper['published'] else "n.d."
                    title = paper['title']
                    pdf_url = paper.get('pdf_url', '')
                    reference = f"{authors} ({year}). *{title}*. {pdf_url}"
                    references.append(reference)

                references_section = "\n\n## References\n" + "\n".join(
                    f"{i + 1}. {ref}" for i, ref in enumerate(references))

                full_summary = summary + references_section

                with summary_container:
                    st.markdown("## Research Summary")
                    st.markdown(full_summary)
                    st.download_button(
                        label="Download Summary",
                        data=full_summary,
                        file_name="research_summary.md",
                        mime="text/markdown",
                    )

    with tab3:
        st.subheader("Evaluate Summaries")

        if 'similar_papers' not in st.session_state:
            st.warning("Please find similar papers and generate summaries first.")
        elif 'summary_api' not in st.session_state or 'summary_ollama' not in st.session_state:
            st.warning("Please generate both OpenAI and Ollama summaries to proceed with evaluation.")
        else:
            paper_ids = [paper['id'] for paper, _ in st.session_state['similar_papers']]
            query = st.session_state['user_query']

            with st.spinner("Evaluating summaries using ROUGE metrics..."):
                df, comparison_chart, radar_chart = evaluate_rouge.evaluate_summaries(
                    paper_ids,
                    query,
                    st.session_state['summary_api'],
                    st.session_state['summary_ollama']
                )

            st.markdown("### ROUGE Score Table")
            st.dataframe(df)

            st.markdown("### Score Comparison Chart")
            st.image(comparison_chart)

            st.markdown("### Radar Chart")
            st.image(radar_chart)

            if st.button("💾 Save Evaluation Results"):
                path = evaluate_rouge.save_evaluation_results(
                    query,
                    evaluate_rouge.create_reference_text(evaluate_rouge.get_paper_details(paper_ids)),
                    st.session_state['summary_api'],
                    st.session_state['summary_ollama'],
                    df,
                    comparison_chart,
                    radar_chart
                )
                st.success(f"Evaluation results saved to: {path}")

if __name__ == "__main__":
    main()