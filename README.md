# Wikipedia-based Document Classifier
Wikipedia-based Document Classifier is a module for automatic multilabel classification of English texts using Wikipedia pages.

## Install
- Using pip:
```
pip install wiki_based_doc_classifier-0.1.0-py3-none-any.whl
```

## Usage
In the following example we use [Wikipedia-API](https://github.com/martin-majlis/Wikipedia-API) to retrieve some pages from English Wikipedia.

[example_usage.py](examples/example_usage.py)
```python
import wikipediaapi
from wiki_based_doc_classifier.classifiers import TfidfClassifier

wiki_wiki = wikipediaapi.Wikipedia('en')

page_titles = ['Farming', 'Art', 'Koala', 'Pablo Picasso', 'Prehistoric Art', 'Cubism']
page_texts = [wiki_wiki.page(t).text for t in page_titles]

classifier = TfidfClassifier(num_labels=2)
classifier.fit(page_texts, page_titles)

predictions = classifier.predict(['Pablo Picasso was one of the greatest painters of the XX century.'])
print(predictions)
```
Output:
```
[['Pablo Picasso', 'Cubism', 'Art']]
```




