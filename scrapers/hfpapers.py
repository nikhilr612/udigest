# Scrape papers from HF.
# Using beautifulsoup4 to parse HTML content, since no official API is available.
# Maybe consider using 3rd-party Feeds.

from bs4 import BeautifulSoup
import requests
from scrapers import DetailScraper, jenv as env
import json

template = env.get_template('hfpaper_item.jinja2')

class HFPapersScraper(DetailScraper):
    def __init__(self, target_url: str = "https://huggingface.co/papers") -> None:
        DetailScraper.__init__(self,target_url)

    def scrape(self) -> list[str]:
        """
        Scrape and return a list of strings with paper details from Hugging Face papers page.
        
        :return: List of paper titles.
        :rtype: list[str]
        """
        url = self.target_url
        response = requests.get(url)
        response.raise_for_status()  # Ensure we got a successful response

        # print(response.text)
        soup = BeautifulSoup(response.text, 'html.parser')
        divs = soup.select("div.SVELTE_HYDRATER.contents[data-target=\"DailyPapers\"]")

        paper_dicts = [json.loads(div.get('data-props')) for div in divs]

        # Important fields
        # id, authors.name, publishedAt, title, summary, githubRepo, upvotes, organization.name
        # jinja template can format these details nicely.
        paper_details = []
        for paper in paper_dicts[0].get('dailyPapers', []):
            paper_info = paper.get('paper', {})
            rendered = template.render(paper=paper_info)
            paper_details.append(rendered)

        return paper_details