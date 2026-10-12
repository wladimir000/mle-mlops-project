import unittest

from fastapi.testclient import TestClient

from src.api import app


class TestTaxiAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_health(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_loaded"])

    def test_prediction(self):
        payload = {
            "PULocationID": 74,
            "DOLocationID": 236,
            "trip_distance": 3.5,
        }

        response = self.client.post(
            "/predict", json=payload
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "predicted_duration_minutes",
            response.json()
        )
        self.assertIsInstance(
            response.json()["predicted_duration_minutes"],
            float
        )

    def test_negative_distance(self):
        payload = {
            "PULocationID": 74,
            "DOLocationID": 236,
            "trip_distance": -1,
        }

        response = self.client.post(
            "/predict", json=payload
        )
        self.assertEqual(response.status_code, 422)

    def test_missing_features(self):
        response = self.client.post(
            "/predict", json={"trip_distance": 3.5}
        )
        self.assertEqual(response.status_code, 422)

    def test_unexpected_features(self):
        payload = {
            "PULocationID": 74,
            "DOLocationID": 236,
            "trip_distance": 3.5,
            "unknown_feature": 123,
        }

        response = self.client.post(
            "/predict", json=payload
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
