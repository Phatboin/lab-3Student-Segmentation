import unittest
import numpy as np
from src.models.kmeans import KMeansNumPy

class TestKMeans(unittest.TestCase):
    def test_fit_predict(self):
        X = np.array([[10, 10], [12, 11], [85, 88], [90, 92]], dtype=float)
        model = KMeansNumPy(n_clusters=2, random_state=42).fit(X)
        preds = model.predict(X)
        self.assertEqual(len(preds), 4)
        self.assertEqual(preds[0], preds[1])
        self.assertEqual(preds[2], preds[3])
        self.assertNotEqual(preds[0], preds[2])

if __name__ == "__main__":
    unittest.main()
