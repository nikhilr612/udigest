"""
Generic interface for detail scrapers.
"""

from jinja2 import Environment, FileSystemLoader
from abc import ABC, abstractmethod

jenv = Environment(loader=FileSystemLoader('templates'))

class DetailScraper(ABC):
    target_url: str

    def __init__(self, target_url: str) -> None:
        self.target_url = target_url

    @abstractmethod
    def scrape(self) -> list[str]:
        """
        Scrape and return a list of text.
        
        :param self: The instance of the scraper.
        :return: List of strings with relavant text (eg. paper metadata)
        :rtype: list[str]
        """
        pass