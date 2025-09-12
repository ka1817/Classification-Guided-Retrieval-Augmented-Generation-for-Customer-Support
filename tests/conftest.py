import os
import pandas as pd
import pytest

TEST_CSV_PATH = "tests/test_data.csv"

@pytest.fixture(scope="session", autouse=True)
def setup_test_csv():
    df = pd.DataFrame({
        "query": [
            "What are the side effects of the COVID-19 vaccine?",
            "How can I check my account balance?",
        ],
        "response": [
            "Common side effects of the COVID-19 vaccine include soreness at the injection site, fever, and fatigue.",
            "You can check your balance by logging into your account online or using our mobile app.",
        ],
        "intent": [
            "side effects inquiry",
            "balance inquiry",
        ],
        "domain": [
            "healthcare",
            "finance",
        ],
    })
    df.to_csv(TEST_CSV_PATH, index=False)
    yield
    os.remove(TEST_CSV_PATH)

@pytest.fixture
def test_csv_path():
    return TEST_CSV_PATH
