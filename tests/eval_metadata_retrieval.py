from dataclasses import replace
from pathlib import Path
import yaml
import json

from src.config import RAGConfig
from src.retriever import load_artifacts, FAISSRetriever, BM25Retriever, IndexKeywordRetriever, filter_retrieved_chunks
from src.ranking.ranker import EnsembleRanker

from typing import Dict, Any, TypedDict

INDEX_PREFIX = "textbook_index"

ChunkID = int
Score = float
MetadataRecord = dict[str, Any]

class RetrievalResult(TypedDict):
    selected_chunk_ids: list[ChunkID]
    selected_scores: list[Score]
    selected_sections: list[str]
    fused_chunk_ids: list[ChunkID]

class BenchmarkResult(TypedDict):
    benchmark_id: str
    question: str
    level: int
    selected_chunk_ids: list[int]
    selected_scores: list[float]
    selected_sections: list[str]
    ideal_retrieved_chunks: list[int]
    section_coverage: int
    ground_truth_in_k: float
    notes: str
    ground_truth_in_fused: float

class ConfigEvaluationResult(TypedDict):
    label: str
    avg_section_coverage: list[float]
    avg_ground_truth_in_k: list[float]
    results: list[list[BenchmarkResult]]

def load_metadata_benchmarks(path: str | Path) -> list[dict]:
    path = Path(path)

    with path.open("r") as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, dict):
        raise ValueError("Benchmarks file must contain a top-level mapping")

    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError("Benchmarks file must contain a 'benchmark' list.")
    
    validated = []
    for i, item in enumerate(benchmarks):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark at index {i} must be a mapping.")

        benchmark_id = item.get("id")
        question = item.get("question")
        ideal_chunks = item.get("ideal_retrieved_chunks")

        if not isinstance(benchmark_id, str) or not benchmark_id.strip():
            raise ValueError(f"Benchmark at index {i} does not have valid id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Benchmark at index {i} does not have a valid question")
        if not isinstance(ideal_chunks, list) or not all(isinstance(x, int) for x in ideal_chunks):
            raise ValueError(f"Benchmark {benchmark_id} must have valid list of ints for ideal chunks")
        
        validated.append(item)
    return validated

def build_retrieval_pipeline(cfg: RAGConfig):
    artifacts_dir = cfg.get_artifacts_directory()
    faiss_idx, bm25_idx, chunks, sources, metadata = load_artifacts(artifacts_dir, INDEX_PREFIX)

    retrievers = [FAISSRetriever(faiss_idx, cfg.embed_model), BM25Retriever(bm25_idx),]
    if cfg.ranker_weights.get("index_keywords", 0) > 0:
        retrievers.append(IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path))
    
    ranker = EnsembleRanker(cfg.ensemble_method, cfg.ranker_weights, int(cfg.rrf_k))

    return chunks, metadata, retrievers, ranker

def retrieve_chunks_for_query(question: str, cfg: RAGConfig, chunks: list[str], metadata: list[MetadataRecord], retrievers: list[Any], ranker: EnsembleRanker) -> RetrievalResult:
    raw_scores: Dict[str, Dict[int, float]] = {}
    for retriever in retrievers:
        raw_scores[retriever.name] = retriever.get_scores(question, cfg.num_candidates, chunks)
    ordered_ids, ordered_scores = ranker.rank(raw_scores=raw_scores)
    selector_ids = ordered_ids[:cfg.selector_pool_size]
    selector_scores = ordered_scores[:cfg.selector_pool_size]
    topk_idxs, selected_scores = filter_retrieved_chunks(cfg, chunks, selector_ids, selector_scores, metadata)

    selected_sections = [
        metadata[idx].get("section_path", "")
        for idx in topk_idxs
        if 0 <= idx < len(metadata)
    ]

    return {
        "selected_chunk_ids": topk_idxs,
        "selected_scores": selected_scores,
        "selected_sections": selected_sections,
        "fused_chunk_ids": ordered_ids
    }


def compute_section_coverage(selected_sections: list[str]) -> int:
    sections_normalized = [section for section in selected_sections if section]
    return len(set(sections_normalized))

def compute_ground_truth_in_k(selected_ids: list[int], ideal_ids: list[int]) -> float:
    unique_selected = set(selected_ids)
    unique_ideal = set(ideal_ids)
    if not unique_ideal:
        return 0.0
    intersection_size = len(unique_selected & unique_ideal)
    return intersection_size / len(unique_ideal)

def compute_ground_truth_in_fused(fused_chunk_ids: list[ChunkID], ideal_ids: list[int]) -> float:
    unique_fused = set(fused_chunk_ids)
    unique_ideal = set(ideal_ids)
    if not unique_ideal:
        return 0.0
    intersection_size = len(unique_ideal & unique_fused)
    return intersection_size / len(unique_ideal)

def evaluate_config(benchmarks: list[dict[str, Any]], cfg: RAGConfig, label: str) -> ConfigEvaluationResult:
    chunks, metadata, retrievers, ranker = build_retrieval_pipeline(cfg=cfg)
    results: list[list[BenchmarkResult]] = [[], [], []]

    for benchmark in benchmarks:
        question = benchmark["question"]
        ideal_ids = benchmark["ideal_retrieved_chunks"]
        level = benchmark["level"]

        retrieval_result = retrieve_chunks_for_query(
            question=question,
            cfg=cfg,
            chunks=chunks,
            metadata=metadata,
            retrievers=retrievers,
            ranker=ranker
        )

        section_coverage = compute_section_coverage(retrieval_result["selected_sections"])
        ground_truth_in_k = compute_ground_truth_in_k(retrieval_result["selected_chunk_ids"], ideal_ids)
        ground_truth_in_fused = compute_ground_truth_in_fused(retrieval_result["fused_chunk_ids"], ideal_ids)

        results[level-1].append({
            "benchmark_id": benchmark["id"],
            "question": question,
            "level": level,
            "selected_chunk_ids": retrieval_result["selected_chunk_ids"],
            "selected_scores": retrieval_result["selected_scores"],
            "selected_sections": retrieval_result["selected_sections"],
            "ideal_retrieved_chunks": ideal_ids,
            "section_coverage": section_coverage,
            "ground_truth_in_k": ground_truth_in_k,
            "notes": benchmark.get("notes", ""),
            "ground_truth_in_fused": ground_truth_in_fused
        })
    
    if not results:
        return {
            "label": label,
            "avg_section_coverage": [0.0, 0.0, 0.0],
            "avg_ground_truth_in_k": [0.0, 0.0, 0.0],
            "results": []
        }
    
    avg_section_coverage = [0.0, 0.0, 0.0]
    avg_ground_truth_in_k = [0.0, 0.0, 0.0]
    for idx, level_result in enumerate(results):
        avg_section_coverage[idx] = sum(result["section_coverage"] for result in level_result) / len(level_result)
        avg_ground_truth_in_k[idx] = sum(result["ground_truth_in_k"] for result in level_result) / len(level_result)

    return {
        "label": label,
        "avg_section_coverage": avg_section_coverage,
        "avg_ground_truth_in_k": avg_ground_truth_in_k,
        "results": results
    }


def main():
    benchmark_path = Path("tests/metadata-retrieval-benchmarks.yaml")
    benchmarks = load_metadata_benchmarks(benchmark_path)

    base_cfg = RAGConfig()

    eval_configs = [
        #---------------------------------------------------- BASELINE --------------------------------------------------------------#
        ("baseline_pool20", replace(base_cfg, use_section_diversity=False, selector_pool_size=20)),
        #---------------------------------------------------- SECTION_DIVERSITY -----------------------------------------------------#
        ("max_chunks1_pool15", replace(base_cfg, use_section_diversity=True, max_chunks_per_section=1, selector_pool_size=15)),
        ("max_chunks1_pool20", replace(base_cfg, use_section_diversity=True, max_chunks_per_section=1, selector_pool_size=20)),
        ("max_chunks1_pool30", replace(base_cfg, use_section_diversity=True, max_chunks_per_section=1, selector_pool_size=30)),
        ("max_chunks2_pool20", replace(base_cfg, use_section_diversity=True, max_chunks_per_section=2, selector_pool_size=20)),
        #---------------------------------------------------- CONTEXT_BOOSTING ------------------------------------------------------#
        ("neigh_boost0.3_N5_pool15", replace(base_cfg, use_context_boosting=True, neighbor_boost=0.3, context_boost_top_N=5, selector_pool_size=15)),
        ("neigh_boost0.3_N5_pool20", replace(base_cfg, use_context_boosting=True, neighbor_boost=0.3, context_boost_top_N=5, selector_pool_size=20)),
        ("neigh_boost0.6_N5_pool15", replace(base_cfg, use_context_boosting=True, neighbor_boost=0.6, context_boost_top_N=5, selector_pool_size=15)),
        ("neigh_boost0.6_N5_pool20", replace(base_cfg, use_context_boosting=True, neighbor_boost=0.6, context_boost_top_N=5, selector_pool_size=20)),
        ("neigh_boost0.3_N10_pool15", replace(base_cfg, use_context_boosting=True, neighbor_boost=0.3, context_boost_top_N=10, selector_pool_size=20)),
        #---------------------------------------------------- PAGE_INDEPENDENCE -----------------------------------------------------#
        ("page_ind_penalty.1_pool15", replace(base_cfg, use_page_independence=True, page_independence_penalty=0.1, selector_pool_size=15)),
        ("page_ind_penalty.1_pool20", replace(base_cfg, use_page_independence=True, page_independence_penalty=0.1, selector_pool_size=20)),
        ("page_ind_penalty.3_pool15", replace(base_cfg, use_page_independence=True, page_independence_penalty=0.3, selector_pool_size=15)),
        ("page_ind_penalty.3_pool20", replace(base_cfg, use_page_independence=True, page_independence_penalty=0.3, selector_pool_size=20)),
        ("page_ind_penalty.1_pool30", replace(base_cfg, use_page_independence=True, page_independence_penalty=0.1, selector_pool_size=30)),
        #---------------------------------------------------- REDUNDANCY_PENALTY ----------------------------------------------------#
        ("redun_penalty.2_treshold.5_pool15", replace(base_cfg, use_redundancy_penalty=True, redundancy_penalty=0.2, redundancy_threshold=0.5, selector_pool_size=15)),
        ("redun_penalty.2_treshold.5_pool20", replace(base_cfg, use_redundancy_penalty=True, redundancy_penalty=0.2, redundancy_threshold=0.5, selector_pool_size=20)),
        ("redun_penalty.4_treshold.5_pool15", replace(base_cfg, use_redundancy_penalty=True, redundancy_penalty=0.4, redundancy_threshold=0.5, selector_pool_size=15)),
        ("redun_penalty.4_treshold.5_pool20", replace(base_cfg, use_redundancy_penalty=True, redundancy_penalty=0.4, redundancy_threshold=0.5, selector_pool_size=20)),
        ("redun_penalty.2_treshold.3_pool15", replace(base_cfg, use_redundancy_penalty=True, redundancy_penalty=0.2, redundancy_threshold=0.3, selector_pool_size=15)),
        ("redun_penalty.2_treshold.7_pool15", replace(base_cfg, use_redundancy_penalty=True, redundancy_penalty=0.2, redundancy_threshold=0.7, selector_pool_size=15)),
        #---------------------------------------------------- ALL CONSTRAINTS ----------------------------------------------------#
        ("all_constraints_balanced", 
         replace(base_cfg, use_section_diversity=True, use_context_boosting=True, use_page_independence=True, use_redundancy_penalty=True, 
                 max_chunks_per_section=1, neighbor_boost=0.2, context_boost_top_N=5, page_independence_penalty=0.1, 
                 redundancy_penalty=0.15, redundancy_threshold=0.5, selector_pool_size=15)),
        ("all_constraints_context_heavy", 
         replace(base_cfg, use_section_diversity=True, use_context_boosting=True, use_page_independence=True, use_redundancy_penalty=True, 
                 max_chunks_per_section=1, neighbor_boost=0.3, context_boost_top_N=8, page_independence_penalty=0.1, 
                 redundancy_penalty=0.15, redundancy_threshold=0.5, selector_pool_size=15)),
        ("all_constraints_diversity_heavy", 
         replace(base_cfg, use_section_diversity=True, use_context_boosting=True, use_page_independence=True, use_redundancy_penalty=True, 
                 max_chunks_per_section=1, neighbor_boost=0.2, context_boost_top_N=5, page_independence_penalty=0.2, 
                 redundancy_penalty=0.25, redundancy_threshold=0.45, selector_pool_size=20)),
        ("all_constraints_recall_friendly", 
         replace(base_cfg, use_section_diversity=True, use_context_boosting=True, use_page_independence=True, use_redundancy_penalty=True, 
                 max_chunks_per_section=2, neighbor_boost=0.2, context_boost_top_N=5, page_independence_penalty=0.05, 
                 redundancy_penalty=0.10, redundancy_threshold=0.60, selector_pool_size=15)),
    ]

    summaries = []
    for label, cfg in eval_configs:
        summaries.append(evaluate_config(benchmarks, cfg, label))

    print("\nMetadata Retrieval Evaluation")
    print("=" * 80)
    for summary in summaries:
        print(f"{summary['label']}")
        for i in range(len(summary['avg_section_coverage'])):
            print("=" * 40, f" Level {i+1} ", "=" * 40)
            print(
                f"avg_section_coverage={summary['avg_section_coverage'][i]:.3f}, "
                f"avg_ground_truth_in_k={summary['avg_ground_truth_in_k'][i]:.3f}"
            )
        #if summary['label'] == "baseline_pool20" or summary['label'] == "max_chunks1_pool15":
            #for result in summary['results']:
                #print(f"Results for {summary['label']}:")
                #print("=" * 80)
                #print(f"BenchmarkID: {result['benchmark_id']} \n"
                      #f"Selected Chunk Ids: {result['selected_chunk_ids']} \n"
                      #f"Ideal Chunk Ids: {result['ideal_retrieved_chunks']} \n"
                      #f"Section Coverage: {result['section_coverage']} \n"
                      #f"Ground Truth in k: {result['ground_truth_in_k']} \n"
                      #f"Ground Truth in Fused: {result['ground_truth_in_fused']}")
                
    results_dir = Path("tests/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "metadata_retrieval_results.json"
    with output_path.open("w") as f:
       payload = {
           "benchmark_file": str(benchmark_path),
           "index_prefix": INDEX_PREFIX,
           "summary": summaries
       }
       json.dump(payload, f, indent=2)

    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    main()