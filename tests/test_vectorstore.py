import os
import pytest
from src.vectorstore_builder import VectorStoreBuilder

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstores")

@pytest.fixture(scope="module")
def builder() -> VectorStoreBuilder:
    return VectorStoreBuilder(vectorstore_dir=VECTORSTORE_DIR)


def test_vectorstore_directory_exists(builder):
    assert os.path.isdir(builder.vectorstore_dir), f"❌ Vectorstore directory not found: {builder.vectorstore_dir}"


def test_vectorstore_domain_folders_exist(builder):
    domain_folders = [
        d for d in os.listdir(builder.vectorstore_dir)
        if os.path.isdir(os.path.join(builder.vectorstore_dir, d))
    ]
    assert domain_folders, "❌ No domain vectorstore folders found inside vectorstores/"


def test_load_existing_vectorstore(builder):
    domain_folders = [
        d for d in os.listdir(builder.vectorstore_dir)
        if os.path.isdir(os.path.join(builder.vectorstore_dir, d))
    ]
    if not domain_folders:
        pytest.fail("❌ No vectorstore domain available to load.")

    sample_domain = domain_folders[0]
    vs = builder.load(sample_domain)
    assert vs is not None, f"❌ Failed to load vectorstore for domain: {sample_domain}"
