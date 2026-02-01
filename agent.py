"""
A simple ReAct agent to briefly evaluate Hugging Face papers.
"""

from io import TextIOWrapper
import time
from typing import Self
import arxiv
import wikipedia
from ddgs import DDGS
from tqdm import tqdm
from scrapers import DetailScraper, jenv
import dspy

client = arxiv.Client()
arxiv_template = jenv.get_template('arxiv_item.jinja2')
trajectory_template = jenv.get_template('trajectory.jinja2')

def _arxiv_generic_search(query: str, k: int, sort_criterion: arxiv.SortCriterion, sort_order: arxiv.SortOrder) -> list[str]:
    search = arxiv.Search(
        query=query,
        max_results=k,
        sort_by=sort_criterion,
        sort_order=sort_order,
    )

    results = []
    for res in client.results(search):
        arxiv_text = arxiv_template.render(arxiv=res)
        results.append(arxiv_text)
        
    return results

# Agent Tools

def arxiv_fetch_most_recent(query: str, k: int = 10) -> list[str]:
    """
    Fetch the most recent k papers from arXiv matching the query.
    
    :param query: The search query string.
    :param k: Number of papers to fetch. Default is 10.
    :return: List of text details of the papers.
    :rtype: list[str]
    """
    return _arxiv_generic_search(query, k, arxiv.SortCriterion.SubmittedDate, arxiv.SortOrder.Descending)

def arxiv_fetch_most_relevant(query: str, k: int = 10) -> list[str]:
    """
    Fetch the most relevant k papers from arXiv matching the query.
    
    :param query: The search query string.
    :param k: Number of papers to fetch. Default is 10.
    :return: List of text details of the papers.
    :rtype: list[str]
    """
    return _arxiv_generic_search(query, k, arxiv.SortCriterion.Relevance, arxiv.SortOrder.Descending)

def wikipedia_term_search(term: str, k: int = 5) -> list[str]:
    """
    Fetch the top k Wikipedia search results for the given term.
    
    :param term: The search term.
    :param k: The maximum number of results to fetch. Default is 5.
    :return: List of Wikipedia article summaries.
    :rtype: list[str]
    """
    # Query wikipedia for the term
    results = []
    disamb_info = None
    
    try:
        # Try to get the page summary directly
        summary = wikipedia.summary(term)
        results.append(summary)
    except wikipedia.exceptions.DisambiguationError as e:
        # Store disambiguation info to include with summaries
        disamb_info = f"Disambiguation for '{term}':\n" + "\n".join(e.options[:k])
        
        # Query each disambiguation option for summaries
        for option in e.options[:k]:
            try:
                summary = wikipedia.summary(option)
                # Include disambiguation info with the first successful summary
                if disamb_info:
                    results.append(f"{disamb_info}\n\n{summary}")
                    disamb_info = None
                else:
                    results.append(summary)
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                # If option also causes issues, skip it
                pass
    except wikipedia.exceptions.PageError:
        # Page not found
        results.append(f"No Wikipedia page found for '{term}'")
    
    # If we have disambiguation info but no successful summaries, return it as single line
    if disamb_info:
        results.append(disamb_info)
    
    return results[:k] 
    
def generic_internet_term_search(term: str, k: int = 5) -> list[str]:
    """
    Perform a web search for the given term using a generic search engine.
    Returns the top k instant results. Snippets are limited to 1024 characters.
    
    :param term: The search term.
    :param k: The maximum number of results to fetch. Default is 5.
    :return: List of search result snippets.
    :rtype: list[str]
    """
    results = []
    with DDGS() as ddgs:
        search_results = ddgs.text(term, max_results=k)
        for result in search_results:
            snippet = f"Link: {result.get('href')}\nText: {result.get('body').strip()[:1024]}\n"
            results.append(snippet)
    return results

class PaperCurationService(dspy.Signature):
    """
    You are a paper curation agent that evaluates and curates papers based on their summaries according to user preferences and input.
    You are provided with tools to search for similar papers, fetch wikipedia information and perform web searches.
    You are required to perform a brief survey of related literature, and provide remarks in your final answer.
    """
    user_preference: str = dspy.InputField(desc="User preferences for paper curation, e.g., focus areas, methodologies, etc")
    paper_information: str = dspy.InputField(desc="The paper information text to evaluate and curate.")
    paper_decision: bool = dspy.OutputField(desc="Whether the paper is relevant to the user's preferences.")
    paper_remarks: str = dspy.OutputField(desc="Brief remarks on the paper, including related literature survey and evaluation.")


def new_instance(max_iters: int = 5) -> dspy.ReAct:
    """
    Make a new ReAct agent instance.
    :param max_iters: Maximum iterations for the agent. Default is 5.
    :return: A ReAct agent instance configured for paper curation.
    """
    return dspy.ReAct(
        PaperCurationService,
        max_iters=5,
        tools=[
            arxiv_fetch_most_recent, arxiv_fetch_most_relevant, 
            wikipedia_term_search, 
            generic_internet_term_search,
        ],
    )

class TaskScaffold:
    """
    A class to handle paper curation task, given an agent.
    """
    user_prefs: str = ""
    output_path: str
    agent: dspy.ReAct
    output_file: TextIOWrapper | None = None
    log_trajectory: bool = False
    trajectory_file: TextIOWrapper | None = None

    def __init__(self, user_prefs_path: str, output_path: str, agent: dspy.ReAct, log_trajectory: bool = False) -> None:
        with open(user_prefs_path, 'r', encoding='utf-8') as f:
            self.user_prefs = f.read()
        self.output_path = output_path
        self.output_file = None
        self.agent = agent
        self.log_trajectory = log_trajectory
        self.trajectory_file = None

    def __enter__(self) -> Self:
        """Open the output file and trajectory file (if enabled) for writing."""
        self.output_file = open(self.output_path, 'w', encoding='utf-8')
        if self.log_trajectory:
            trajectory_path = self.output_path.rsplit('.', 1)[0] + '_trajectory.md'
            self.trajectory_file = open(trajectory_path, 'w', encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the output files and free up resources."""
        if self.output_file:
            self.output_file.close()
            self.output_file = None
        if self.trajectory_file:
            self.trajectory_file.close()
            self.trajectory_file = None

    def curate(self, papers: DetailScraper) -> None:
        """
        Curate papers using the agent and write results to output file.
        Each paper is fed serially, as opposed to batch mode.
        :param papers: An instance of DetailScraper to fetch paper details.
        :return: None
        """
        if self.output_file is None:
            raise RuntimeError("TaskScaffold must be used as a context manager (with statement)")

        paper_texts = papers.scrape()
        for idx, paper_text in tqdm(enumerate(paper_texts), total=len(paper_texts), desc="Curating papers"):
            result: dspy.Prediction = self.agent(
                user_preference=self.user_prefs,
                paper_information=paper_text,
            )
            
            # Log trajectory if enabled
            if self.log_trajectory and self.trajectory_file and hasattr(result, 'trajectory'):
                num_steps = sum(1 for key in result.trajectory.keys() if key.startswith('thought_'))
                trajectory_md = trajectory_template.render(
                    trajectory=result.trajectory,
                    paper_idx=idx,
                    total_papers=len(paper_texts),
                    final_decision=result.paper_decision,
                    final_remarks=result.paper_remarks,
                    num_steps=num_steps
                )
                self.trajectory_file.write(trajectory_md)
                self.trajectory_file.write("\n\n")
                self.trajectory_file.flush()
            
            if result.paper_decision:
                self.output_file.write(f"# Paper {idx+1} / {len(paper_texts)}\n")
                self.output_file.write(f"## Information:\n{paper_text}\n\n")
                self.output_file.write(f"## Remarks:\n{result.paper_remarks}\n")
                self.output_file.write("\n\n")
            print("debug:", result)
        print(f"Curation completed. Results saved to {self.output_path}")