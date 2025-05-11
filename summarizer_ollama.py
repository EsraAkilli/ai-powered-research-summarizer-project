import requests
import json
import logging
from typing import List, Dict, Any, Optional, Callable

from summarizer import load_config, create_summary_prompt

# Get logger from main module
logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

def setup_ollama():
    """Setup Ollama configuration."""
    config = load_config()
    ollama_config = config.get('ollama', {})

    # Get values from config or use defaults
    base_url = ollama_config.get('base_url', 'http://localhost:11434')
    model = ollama_config.get('model', 'llama3')

    logger.info(f"Using Ollama with model: {model} at {base_url}")
    return base_url, model

def generate_summary_ollama(papers_content: List[Dict[str, Any]], user_query: str,
                           progress_callback: Optional[ProgressCallback] = None) -> str:
    """Generate a summary of the papers using Ollama API."""
    if progress_callback:
        progress_callback(0.6, "Preparing prompt for Ollama...")

    prompt = create_summary_prompt(papers_content, user_query)

    # Setup Ollama
    if progress_callback:
        progress_callback(0.7, "Setting up Ollama...")

    try:
        base_url, model = setup_ollama()

        # Send request to Ollama API
        if progress_callback:
            progress_callback(0.8, "Sending request to Ollama...")

        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 4000
        }

        # Make API request
        response = requests.post(
            f"{base_url}/api/generate",
            headers=headers,
            data=json.dumps(data),
            timeout=300  # Long timeout for summary generation
        )

        # Check response
        if response.status_code != 200:
            error_msg = f"Ollama API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(1.0, f"Error: {error_msg}")
            return error_msg

        # Process response
        result = response.json()
        summary = result.get('response', '')

        if progress_callback:
            progress_callback(0.9, "Processing response...")
            progress_callback(1.0, "Summary generated successfully!")

        return summary

    except Exception as e:
        error_msg = f"Error generating summary with Ollama: {str(e)}"
        logger.error(error_msg)
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        return error_msg