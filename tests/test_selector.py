from src.config import RAGConfig
from src.retriever import filter_retrieved_chunks


def test_filter_retrieved_chunks_returns_plain_top_k_when_all_constraints_disabled():
    cfg = RAGConfig(
        top_k=3,
        num_candidates=6,
        selector_pool_size=6,
        use_section_diversity=False,
        use_context_boosting=False,
        use_page_independence=False,
        use_redundancy_penalty=False,
    )

    chunks = [f"chunk {i}" for i in range(6)]
    ordered_ids = [0, 1, 2, 3, 4, 5]
    ordered_scores = [0.95, 0.92, 0.90, 0.88, 0.85, 0.80]
    metadata = [{"section_path": f"Section {i}", "page_numbers": [i + 1]} for i in range(6)]

    topk_ids, topk_scores = filter_retrieved_chunks(
        cfg, chunks, ordered_ids, ordered_scores, metadata
    )

    assert topk_ids == [0, 1, 2]
    assert topk_scores == [0.95, 0.92, 0.90]


def test_filter_retrieved_chunks_caps_chunks_per_section_when_diversity_enabled():
    cfg = RAGConfig(
        top_k=3,
        num_candidates=6,
        selector_pool_size=6,
        use_section_diversity=True,
        max_chunks_per_section=1,
        use_context_boosting=False,
        use_page_independence=False,
        use_redundancy_penalty=False,
    )

    chunks = [f"chunk {i}" for i in range(6)]
    ordered_ids = [0, 1, 2, 3, 4, 5]
    ordered_scores = [0.95, 0.92, 0.90, 0.88, 0.85, 0.80]
    metadata = [
        {"section_path": "Chapter 1 A", "page_numbers": [1]},
        {"section_path": "Chapter 1 A", "page_numbers": [2]},
        {"section_path": "Chapter 1 B", "page_numbers": [3]},
        {"section_path": "Chapter 1 C", "page_numbers": [4]},
        {"section_path": "Chapter 1 C", "page_numbers": [5]},
        {"section_path": "Chapter 1 D", "page_numbers": [6]},
    ]

    topk_ids, topk_scores = filter_retrieved_chunks(
        cfg, chunks, ordered_ids, ordered_scores, metadata
    )

    assert topk_ids == [0, 2, 3]
    assert topk_scores == [0.95, 0.90, 0.88]


def test_filter_retrieved_chunks_backfills_when_section_cap_blocks_top_k():
    cfg = RAGConfig(
        top_k=3,
        num_candidates=4,
        selector_pool_size=4,
        use_section_diversity=True,
        max_chunks_per_section=1,
        use_context_boosting=False,
        use_page_independence=False,
        use_redundancy_penalty=False,
    )

    chunks = [f"chunk {i}" for i in range(4)]
    ordered_ids = [0, 1, 2, 3]
    ordered_scores = [0.90, 0.80, 0.70, 0.60]
    metadata = [
        {"section_path": "A", "page_numbers": [1]},
        {"section_path": "A", "page_numbers": [2]},
        {"section_path": "A", "page_numbers": [3]},
        {"section_path": "B", "page_numbers": [4]},
    ]

    topk_ids, topk_scores = filter_retrieved_chunks(
        cfg, chunks, ordered_ids, ordered_scores, metadata
    )

    assert len(topk_ids) == 3
    assert len(topk_scores) == 3
    assert topk_ids[0] == 0
    assert 3 in topk_ids


def test_filter_retrieved_chunks_penalizes_same_page_candidates():
    cfg = RAGConfig(
        top_k=2,
        num_candidates=3,
        selector_pool_size=3,
        use_section_diversity=False,
        use_context_boosting=False,
        use_page_independence=True,
        page_independence_penalty=0.2,
        use_redundancy_penalty=False,
    )

    chunks = ["chunk 0", "chunk 1", "chunk 2"]
    ordered_ids = [0, 1, 2]
    ordered_scores = [1.0, 0.95, 0.90]
    metadata = [
        {"section_path": "A", "page_numbers": [1]},
        {"section_path": "B", "page_numbers": [1]},
        {"section_path": "C", "page_numbers": [2]},
    ]

    topk_ids, topk_scores = filter_retrieved_chunks(
        cfg, chunks, ordered_ids, ordered_scores, metadata
    )

    assert topk_ids == [0, 2]
    assert topk_scores == [1.0, 0.90]


def test_filter_retrieved_chunks_penalizes_redundant_chunks():
    cfg = RAGConfig(
        top_k=2,
        num_candidates=3,
        selector_pool_size=3,
        use_section_diversity=False,
        use_context_boosting=False,
        use_page_independence=False,
        use_redundancy_penalty=True,
        redundancy_penalty=0.2,
        redundancy_threshold=0.5,
    )

    chunks = [
        "database transaction lock serializable",
        "database transaction lock serializable conflict",
        "query optimizer cost plan",
    ]
    ordered_ids = [0, 1, 2]
    ordered_scores = [1.0, 0.95, 0.90]
    metadata = [
        {"section_path": "A", "page_numbers": [1]},
        {"section_path": "B", "page_numbers": [2]},
        {"section_path": "C", "page_numbers": [3]},
    ]

    topk_ids, topk_scores = filter_retrieved_chunks(
        cfg, chunks, ordered_ids, ordered_scores, metadata
    )

    assert topk_ids == [0, 2]
    assert topk_scores == [1.0, 0.90]
