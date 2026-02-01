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
    model: str = typer.Option("gemini/gemini-2.5-flash", "--model", "-m", help="LLM model to use"),
    custom_provider: str | None = typer.Option(None, "--provider", "-p", help="Custom LLM provider if any"),
    scraper: ScraperName = typer.Option(ScraperName.HF_PAPERS, "--scraper", "-s", help="Scraper to use"),
    url: str = typer.Option("https://huggingface.co/papers", "--url", "-u", help="URL to scrape"),
    prefs_file: str = typer.Option("./userprefs.txt", "--prefs", "-i", help="User preferences file"),
    output_file: str = typer.Option("./report.md", "--output", "-o", help="Output report file"),
    max_tokens: int | None = typer.Option(None, "--max-tokens", "-t", help="Maximum tokens for LLM responses"),
    max_iters:int = typer.Option(5, "--max-iters", "-n", help="Maximum iterations for the agent"),
    log_trajectory: bool = typer.Option(False, "--log-trajectory", "-l", help="Log agent trajectory (thoughts, actions, observations)")
):
    """
    Curate papers using a ReAct agent based on user preferences.
    """
    if not custom_provider:
        lm = dspy.LM(model, max_tokens=max_tokens)
    else:
        lm = dspy.LM(model, api_base=custom_provider, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    scraper_instance = SCRAPER_REGISTRY[scraper](url)
    with agent.TaskScaffold(prefs_file, output_file, agent.new_instance(max_iters=max_iters), log_trajectory=log_trajectory) as scaffold:
        scaffold.curate(scraper_instance)

if __name__ == "__main__":
    app()
