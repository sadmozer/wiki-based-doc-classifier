from sklearn.model_selection import GridSearchCV

class ParamSearch():
  def __init__(self, classifier, param_grid, refit=False):
    self.classifier = classifier
    self.param_grid = param_grid
    self.refit = refit
    self.results = None
    self.best_params = None
    self._searcher = GridSearchCV(classifier, param_grid, cv=[(slice(None), slice(None))], refit=refit)
  
  def search(self, wikipedia_texts, wikipedia_titles, val_set):
    self._searcher.fit(wikipedia_texts, wikipedia_titles, val_set=val_set)
    self.results = self._searcher.cv_results_
    self.best_params = self._searcher.best_params_
  
