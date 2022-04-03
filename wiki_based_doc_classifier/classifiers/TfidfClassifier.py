from __future__ import print_function
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

class TfidfClassifier(BaseEstimator):
  """ TfidfClassifier implements a multilabel text classifier.
  
    Attributes:
    :param max_features:
        The max number of terms in the vocabulary. This also
        corresponds to dimension of the document
        embeddings. By default it is set to the total number of
        terms present in the training corpus.
    :param num_labels:
    :param radius:
    """

  def __init__(self, max_features=300000,
               num_labels=5,
               radius=None,
               ):
    self.max_features = max_features
    self.num_labels = num_labels
    self.radius = radius

    self._vectorizer = TfidfVectorizer(decode_error='ignore',
                          max_features = self.max_features,
                          lowercase=True,
                          use_idf=True,
                          smooth_idf=False,
                          stop_words='english')
    if(radius):
      self._nn = NearestNeighbors(radius=radius)
    else:
      self._nn = NearestNeighbors(n_neighbors=num_labels)

  def fit(self, wikipedia_texts, wikipedia_titles, val_set=None):
    print(f"Fitting.. {self.get_params()}")
    WDE = self._vectorizer.fit_transform(wikipedia_texts)
    self._nn.fit(WDE)
    self.wikipedia_titles = np.array(wikipedia_titles)

    if(val_set):
      self.val_set = val_set
      self._val_labels_binary = MultiLabelBinarizer(classes=self.wikipedia_titles).fit_transform(self.val_set.labels)
    return self

  def predict(self, documents, return_binary=False):
    embeddings = self._vectorizer.transform(documents)

    if(self.radius):
      # numpy doesnt support array indexing with a variable length array
      indices = self._nn.radius_neighbors(embeddings, radius=self.radius, sort_results=True, return_distance=True)[1]
    else:
      indices = self._nn.kneighbors(embeddings, n_neighbors=self.num_labels, return_distance=False)
    
    if(return_binary):
      predictions_binary = MultiLabelBinarizer(classes=[i for i in range(len(self.wikipedia_titles))]).fit_transform(indices)
      return predictions_binary
    else:
      return [self.wikipedia_titles[i].tolist() for i in indices]
  
  def get_params(self, deep=True):
    return dict(max_features=self.max_features,
                num_labels=self.num_labels,
                radius=self.radius,
                )

  def set_params(self, **parameters):
    if('max_features' in parameters):
      self._vectorizer.max_features = parameters['max_features']
      self.max_features = parameters['max_features']
    if('num_labels' in parameters):
      self.num_labels = parameters['num_labels']
    if('radius' in parameters):
      self.radius = parameters['radius']
    return self
  
  def score(self, y, y_true):
    print("Measuring precision..")
    predictions_binary = self.predict(documents=self.val_set.texts, return_binary=True)
    value = precision_score(self._val_labels_binary, predictions_binary, average='samples', zero_division=0)
    print(f"Score {value}\n")
    return value