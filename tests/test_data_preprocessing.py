from src.data_preprocessing import DataSplitting

def test_chunking_creates_documents():
    splitter = DataSplitting()
    docs = splitter.chunking()
    assert len(docs) > 0
    assert "Query:" in docs[0].page_content
    assert "Response:" in docs[0].page_content

def test_chunk_by_domain_groups_correctly():
    splitter = DataSplitting()
    domain_chunks = splitter.chunk_by_domain()
    assert "healthcare" in domain_chunks
    assert "finance" in domain_chunks
    assert all(len(docs) > 0 for docs in domain_chunks.values())
