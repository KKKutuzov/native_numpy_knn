import numpy as np
from sklearn.base import BaseEstimator
class KNN_classifier(BaseEstimator):
  def __init__(self, n_neighbors: int, **kwargs):
    self.n_neighbors = n_neighbors
    self._fit = False

  def fit(self, x: np.array, y: np.array):
    self.x = x
    self.y = y
    self._fit = True
  
  def mode_one(self, l: np.array):
    vals, counts = np.unique(l, return_counts=True)
    predicted_class = np.argwhere(counts == np.max(counts))
    return vals[predicted_class[0]]

  def predict(self, x: np.array):
    if not self._fit:
      raise Exception("UnfittedError")
    distances = self.compute_distances(x)
    indexes = np.argsort(distances, axis=1)[:, :self.n_neighbors]
    labels_of_top_classes = self.y[indexes]
    predicted_class = np.apply_along_axis(self.mode_one,1,labels_of_top_classes)
    return predicted_class.flatten()
  
  def compute_distances(self, test: np.array):
    g = lambda y: np.apply_along_axis(lambda x: np.linalg.norm(y-x),1,self.x)
    return np.apply_along_axis(g,1,test)

  def score(self,X_test,y_test):
    return np.mean(self.predict(X_test) == y_test)