import json
import os.path
from datasets import load_dataset
from datetime import datetime

class WikipediaCorpus():
    def __init__(self, date=None, language="en"):
        self._wikipedia_corpus = self._load(date, language)
        self.texts = [doc["text"] for doc in self._wikipedia_corpus]
        self.titles = [doc["title"] for doc in self._wikipedia_corpus]

    def _load(self, date, language):
        if not date:
            current_year = str(datetime.now().year).zfill(2)
            current_month = str(datetime.now().month).zfill(2)
            date = f"{current_year}{current_month}01"
        try:
            corpus = load_dataset("wikipedia", f"20200501.{language}", date=date)
        except Exception as e:
            print(e)
        return corpus
