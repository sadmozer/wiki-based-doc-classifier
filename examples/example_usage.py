import wikipediaapi
from wiki_based_doc_classifier.classifiers import TfidfClassifier

wiki_wiki = wikipediaapi.Wikipedia('en')

page_titles = ['Farming', 'Art', 'Koala', 'Pablo Picasso', 'Prehistoric Art', 'Cubism']
page_texts = [wiki_wiki.page(t).text for t in page_titles]

classifier = TfidfClassifier(num_labels=3)
classifier.fit(page_texts, page_titles)

predictions = classifier.predict(['Pablo Picasso was one of the greatest painters of the XX century.'])
print(predictions)
