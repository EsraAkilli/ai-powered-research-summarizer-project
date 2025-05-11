import openai
import logging
from typing import List, Dict, Any, Optional, Callable

from summarizer import load_config, create_summary_prompt

# Get logger from main module
logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

def setup_openai():
    """Setup OpenAI client and get the model name."""
    config = load_config()
    try:
        client = openai.OpenAI(api_key=config.get('openai', {}).get('api_key'))
        model = config.get('openai', {}).get('model', 'gpt-3.5-turbo')
        return client, model
    except KeyError as e:
        logger.error(f"Missing configuration key: {e}")
        raise
    except Exception as e:
        logger.error(f"Error setting up OpenAI: {e}")
        raise

def generate_summary_openai(papers_content: List[Dict[str, Any]], user_query: str,
                           progress_callback: Optional[ProgressCallback] = None) -> str:
    """Generate a summary of the papers using OpenAI API."""
    if progress_callback:
        progress_callback(0.6, "Preparing prompt for OpenAI...")

    prompt = create_summary_prompt(papers_content, user_query)

    if progress_callback:
        progress_callback(0.7, "Sending request to OpenAI...")

    try:
        client, model = setup_openai()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are an expert academic research assistant who creates comprehensive, accurate summaries of scientific papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )

        if progress_callback:
            progress_callback(0.9, "Processing response...")

        summary = response.choices[0].message.content

        if progress_callback:
            progress_callback(1.0, "Summary generated successfully!")

        return summary

    except Exception as e:
        error_msg = f"Error generating summary with OpenAI: {str(e)}"
        logger.error(error_msg)
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        return error_msg