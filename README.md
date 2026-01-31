# udigest
Discover and curate research papers from arXiv and Hugging Face based on your interests.

## Setup
Install dependencies using [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

## Running
1. Edit `userprefs.txt` to set your preferences
2. Add your API KEY. Consult LiteLLM documentation for the environment variables to set.
3. Run with your LLM configuration:
   ```bash
   uv run main.py --model gemini/gemini-2.0-flash
   ```
   Or with a custom provider:
   ```bash
   uv run main.py --model gpt-4 --provider https://api.openai.com/v1
   ```
4. Check `report.txt` for curated results

Use `uv run main.py --help` to see all available options.

## What It Does
- Searches arXiv and Hugging Face Papers for recent papers
- Filters results based on your topic preferences
- Generates a formatted report with the most relevant findings