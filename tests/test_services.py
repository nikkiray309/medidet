import unittest

from config import AppConfig, ConfigurationError
from health_check import check_indexes
from retrieval import RetrievalError, RetrievalErrorCode, RetrievalSuccess, retrieve_image, retrieve_text


class FakeStore:
    def similarity_search(self, query, k):
        return [f"document-{k}"]


class FakeIndex:
    def query(self, **kwargs):
        return {"matches": [{"metadata": {"Disease": "eczema"}}]}


class RetrievalTests(unittest.TestCase):
    def test_text_success_is_typed(self):
        result = retrieve_text("private query", FakeStore())
        self.assertIsInstance(result, RetrievalSuccess)
        self.assertEqual(result.value, ["document-4"])

    def test_image_success_is_typed(self):
        result = retrieve_image([0.0, 1.0], FakeIndex(), expected_dimension=2)
        self.assertIsInstance(result, RetrievalSuccess)
        self.assertEqual(result.value, "eczema")

    def test_dimension_fails_before_provider_call(self):
        result = retrieve_image([0.0], FakeIndex(), expected_dimension=2)
        self.assertEqual(result.code, RetrievalErrorCode.DIMENSION)

    def test_malformed_response_is_actionable(self):
        class EmptyIndex:
            def query(self, **kwargs):
                return {"matches": []}

        result = retrieve_image([0.0], EmptyIndex(), expected_dimension=1)
        self.assertIsInstance(result, RetrievalError)
        self.assertEqual(result.code, RetrievalErrorCode.MALFORMED_RESPONSE)
        self.assertNotIn("matches", result.user_message)

    def test_provider_categories(self):
        for exception, code in (
            (type("AuthenticationError", (Exception,), {})("secret"), RetrievalErrorCode.AUTHENTICATION),
            (TimeoutError("query leaked"), RetrievalErrorCode.TIMEOUT),
            (type("RateLimitError", (Exception,), {})("query leaked"), RetrievalErrorCode.RATE_LIMIT),
            (type("NotFoundException", (Exception,), {})("index not found"), RetrievalErrorCode.MISSING_INDEX),
            (ValueError("vector dimension 3 does not match 4"), RetrievalErrorCode.DIMENSION),
        ):
            class BrokenStore:
                def similarity_search(self, query, k, error=exception):
                    raise error

            self.assertEqual(retrieve_text("private", BrokenStore()).code, code)


class ConfigurationAndHealthTests(unittest.TestCase):
    def setUp(self):
        self.values = {
            "OPENAI_API_KEY": "openai",
            "PINECONE_API_KEY": "pinecone",
            "TEXT_INDEX_NAME": "text",
            "IMAGE_INDEX_NAME": "image",
            "TEXT_EMBEDDING_DIMENSION": "1536",
            "IMAGE_EMBEDDING_DIMENSION": "512",
        }

    def test_config_requires_index_names(self):
        del self.values["IMAGE_INDEX_NAME"]
        with self.assertRaises(ConfigurationError):
            AppConfig.from_mapping(self.values)

    def test_health_checks_index_metadata(self):
        class Client:
            def describe_index(self, name):
                return {"dimension": 1536 if name == "text" else 512}

        healthy, checks = check_indexes(AppConfig.from_mapping(self.values), Client())
        self.assertTrue(healthy)
        self.assertEqual(len(checks), 2)


if __name__ == "__main__":
    unittest.main()
