from enum import Enum
import os
import dspy
import typer

import agent
from scrapers import DetailScraper
from scrapers.hfpapers import HFPapersScraper

class ScraperName(Enum):
    HF_PAPERS = "hf"

SCRAPER_REGISTRY: dict[ScraperName, type[DetailScraper]] = {
    ScraperName.HF_PAPERS: HFPapersScraper,
}

app = typer.Typer()

@app.command()
def main(
    model: str = typer.Option("gemini/gemini-2.0-flash", "--model", "-m", help="LLM model to use"),
    custom_provider: str | None = typer.Option(None, "--provider", "-p", help="Custom LLM provider if any"),
    scraper: ScraperName = typer.Option(ScraperName.HF_PAPERS, "--scraper", "-s", help="Scraper to use"),
    url: str = typer.Option("https://huggingface.co/papers", "--url", "-u", help="URL to scrape"),
    prefs_file: str = typer.Option("./userprefs.txt", "--prefs", "-i", help="User preferences file"),
    output_file: str = typer.Option("./report.txt", "--output", "-o", help="Output report file"),
):
    """
    Curate papers using a ReAct agent based on user preferences.
    """
    if not custom_provider:
        lm = dspy.LM(model)
    else:
        lm = dspy.LM(model, api_base=custom_provider)
    dspy.configure(lm=lm)
    scraper_instance = SCRAPER_REGISTRY[scraper](url)
    with agent.TaskScaffold(prefs_file, output_file, agent.new_instance()) as scaffold:
        scaffold.curate(scraper_instance)

if __name__ == "__main__":
    app()
