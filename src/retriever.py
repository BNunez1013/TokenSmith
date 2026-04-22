"""
retriever.py

Stores core retrieval logic using FAISS and BM25 scoring.
It also contains helpers for loading artifacts and filtering chunks.
"""

from __future__ import annotations

import pathlib
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import nltk
from nltk.stem import WordNetLemmatizer

import faiss
import numpy as np
from src.embedder import CachedEmbedder

from src.config import RAGConfig
from src.index_builder import preprocess_for_bm25


# -------------------------- Embedder cache ------------------------------

_EMBED_CACHE: Dict[str, CachedEmbedder] = {}

def _get_embedder(model_name: str) -> CachedEmbedder:
    if model_name not in _EMBED_CACHE:
        # Use the cached embedding model to avoid reloading it on every call
        _EMBED_CACHE[model_name] = CachedEmbedder(model_name)
    return _EMBED_CACHE[model_name]

# -------------------------- Retriever helpers ------------------------------

def _get_pages(idx: int, metadata: list[dict]) -> set[int]:
    if idx < 0 or idx >= len(metadata):
        return set()

    pages = metadata[idx].get("page_numbers") or []
    return {int(page) for page in pages}

def _select_baseline_top_k(cfg: RAGConfig, ordered_ids: list[int], ordered_scores: list[float]) -> tuple[list[int], list[float]]:
    return ordered_ids[:cfg.top_k], ordered_scores[:cfg.top_k]

def _metadata_selector_enabled(cfg: RAGConfig) -> bool:
    return (
        cfg.use_section_diversity or cfg.use_context_boosting or cfg.use_page_independence or cfg.use_redundancy_penalty
    )

def _apply_context_boosting(cfg: RAGConfig, ordered_ids: list[int], ordered_scores: list[float],
                            objective_scores: dict[int, float]) -> None:
    for idx, score in zip(ordered_ids[:cfg.context_boost_top_N], ordered_scores[:cfg.context_boost_top_N]):
        for neighbor in (idx-1, idx+1):
            if neighbor in objective_scores:
                objective_scores[neighbor] += score * cfg.neighbor_boost

def _jaccard_similarity(x: set[str], y: set[str]) -> float:
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)

def _compute_running_candidate_scores(cfg: RAGConfig, idx: int, base_score: float, metadata: list[dict], selected_pages: set[int],
                                      selected_chunk_sets: list[set[str]], chunks: list[str]) -> float:
    score = base_score

    if cfg.use_page_independence:
        candidate_pages = _get_pages(idx, metadata)
        if candidate_pages & selected_pages:
            score -= score * cfg.page_independence_penalty
    
    if cfg.use_redundancy_penalty:
        candidate_chunk_tokens = set(preprocess_for_bm25(chunks[idx]))
        max_overlap = max((_jaccard_similarity(candidate_chunk_tokens, selected_tokens) for selected_tokens in selected_chunk_sets), default=0.0)
        if max_overlap >= cfg.redundancy_threshold:
            score -= score * cfg.redundancy_penalty

    return score

def _compute_objective_scores(cfg: RAGConfig, ordered_ids: list[int], ordered_scores: list[float],
                              metadata: list[dict], chunks: list[str]) -> dict[int, float]:
    objective_scores = dict(zip(ordered_ids, ordered_scores))

    if cfg.use_context_boosting:
        _apply_context_boosting(cfg, ordered_ids, ordered_scores, objective_scores)

    return objective_scores

def _rank_chunk_candidates(ordered_ids: list[int], objective_scores: dict[int, float]) -> list[int]:
    return sorted(ordered_ids, key=lambda idx: objective_scores.get(idx, 0.0), reverse=True)

def _passes_constraints(cfg: RAGConfig, idx: int, metadata: list[dict], section_counts: dict[str, int]) -> bool:
    if cfg.use_section_diversity:
        section_path = metadata[idx].get("section_path")
        if not section_path:
            section_path = "missing_section"
        if section_counts.get(section_path, 0) >= cfg.max_chunks_per_section:
            return False
    return True

def _greedy_selection_with_constraints(cfg: RAGConfig, ranked_candidates: list[int], objective_score: dict[int, float],
                                       metadata: list[dict], chunks: list[str]) -> tuple[list[int], list[float], set[int], list[set[str]]]:
    selected_ids: list[int] = []
    selected_scores: list[float] = []
    section_counts: dict[str, int] = {}
    selected_pages = set()
    remaining_candidates = list(ranked_candidates)
    selected_chunk_set: list[set[str]] = []

    while remaining_candidates and len(selected_ids) < cfg.top_k:
        best_idx = None
        best_score = float('-inf')

        for idx in remaining_candidates:
            if not _passes_constraints(cfg, idx, metadata, section_counts):
                continue

            score = _compute_running_candidate_scores(cfg, idx, objective_score.get(idx, 0), metadata, selected_pages, selected_chunk_set, chunks)

            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx is None:
            break

        selected_ids.append(best_idx)
        selected_scores.append(best_score)
        remaining_candidates.remove(best_idx)

        if cfg.use_page_independence:
            selected_pages.update(_get_pages(best_idx, metadata))

        if cfg.use_section_diversity:
            section_path = metadata[best_idx].get("section_path")
            if not section_path:
                section_path = "missing_section"
            section_counts[section_path] = section_counts.get(section_path, 0) + 1
        
        if cfg.use_redundancy_penalty:
            selected_chunk_set.append(set(preprocess_for_bm25(chunks[best_idx])))
        
    return selected_ids, selected_scores, selected_pages, selected_chunk_set

def _backfill_selection(cfg: RAGConfig, selected_ids: list[int], selected_scores: list[float],
                        ranked_candidates: list[int], objective_scores: dict[int, float], metadata: list[dict], 
                        selected_pages: set[int], chunks: list[str], selected_chunk_sets: list[set[str]]) -> tuple[list[int], list[float]]:
    selected_set = set(selected_ids)
    remaining = list(ranked_candidates)
    while remaining and len(selected_set) < cfg.top_k:
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            if idx in selected_set:
                continue

            score = _compute_running_candidate_scores(cfg, idx, objective_scores.get(idx, 0), metadata, selected_pages, selected_chunk_sets, chunks)

            if score > best_score:
                best_score = score
                best_idx = idx

        selected_ids.append(best_idx)
        selected_scores.append(best_score)
        selected_set.add(best_idx)
        remaining.remove(best_idx)

        if cfg.use_page_independence:
            selected_pages.update(_get_pages(best_idx, metadata))
        
        if cfg.use_redundancy_penalty:
            selected_chunk_sets.append(set(preprocess_for_bm25(chunks[best_idx])))
    
    return selected_ids, selected_scores
# -------------------------- Read artifacts -------------------------------

def load_artifacts(artifacts_dir: os.PathLike, index_prefix: str) -> Tuple[faiss.Index, List[str], List[str], Any]:
    """
    Loads:
      - FAISS index: {index_prefix}.faiss
      - chunks:      {index_prefix}_chunks.pkl
      - sources:     {index_prefix}_sources.pkl
    """
    artifacts_dir = pathlib.Path(artifacts_dir)
    faiss_index = faiss.read_index(str(artifacts_dir / f"{index_prefix}.faiss"))
    bm25_index  = pickle.load(open(artifacts_dir / f"{index_prefix}_bm25.pkl", "rb"))
    chunks      = pickle.load(open(artifacts_dir / f"{index_prefix}_chunks.pkl", "rb"))
    sources     = pickle.load(open(artifacts_dir / f"{index_prefix}_sources.pkl", "rb"))
    metadata = pickle.load(open(artifacts_dir / f"{index_prefix}_meta.pkl", "rb"))

    return faiss_index, bm25_index, chunks, sources, metadata


# -------------------------- Helper to get page nums for chunks -------------------------------

def get_page_numbers(chunk_indices: list[int], metadata: list[dict]) -> dict[int, List[int]]:
    if not metadata or not chunk_indices:
        return {}

    page_map: dict[int, List[int]] = {}

    for chunk_idx in chunk_indices:
        chunk_idx = int(chunk_idx)
        if 0 <= chunk_idx < len(metadata):
            chunk_pages = metadata[chunk_idx].get("page_numbers")
            if chunk_pages is None:
                continue  # don't store None; callers can default to [1]
            page_map[chunk_idx] = chunk_pages

    return page_map

# -------------------------- Filtering logic -----------------------------

def filter_retrieved_chunks(cfg: RAGConfig, chunks: list[str], ordered_ids: list[int], 
                            ordered_scores: list[float], metadata: list[dict] | None = None, effective_topk: int = 0) -> tuple[list[int], list[float]]:
    
    if effective_topk != 0:
        cfg.top_k = effective_topk
    baseline = _select_baseline_top_k(cfg, ordered_ids, ordered_scores)

    if metadata is None or len(ordered_ids) != len(ordered_scores):
        return baseline
    
    if not _metadata_selector_enabled(cfg):
        return baseline
    
    objective_scores = _compute_objective_scores(cfg, ordered_ids, ordered_scores, metadata, chunks)
    ranked_candidates = _rank_chunk_candidates(ordered_ids, objective_scores)
    selected_ids, selected_scores, selected_pages, selected_chunk_sets = _greedy_selection_with_constraints(cfg, ranked_candidates, objective_scores, metadata, chunks)

    return _backfill_selection(cfg, selected_ids, selected_scores, ranked_candidates, objective_scores, metadata, selected_pages, chunks, selected_chunk_sets)

# -------------------------- Retrieval core ------------------------------

class Retriever(ABC):
    @abstractmethod
    def get_scores(self, query: str, pool_size: int, chunks: List[str]):
        """Retrieves the top 'pool_size' chunks cores for a given query."""
        pass


class FAISSRetriever(Retriever):
    name = "faiss"

    def __init__(self, index, embed_model: str):
        self.index = index
        self.embedder = _get_embedder(embed_model)

    def get_scores(self,
                query: str,
                pool_size: int,
                chunks: List[str]) -> Dict[int, float]:
        """
        Returns FAISS scores for top 'pool_size' keyed by global chunk index.
        """
        # FAISS expects a 2D array
        q_vec = self.embedder.encode([query]).astype("float32")
        
        # Safety check on vector dimensions
        if q_vec.shape[1] !=  self.index.d:
            raise ValueError(
                f"Embedding dim mismatch: index={ self.index.d} vs query={q_vec.shape[1]}"
            )

        # Perform the search
        distances, indices =  self.index.search(q_vec, pool_size)

        # Remove invalid indices and ensure they are within bounds
        cand_idxs = [i for i in indices[0] if 0 <= i < len(chunks)]

        # Create the distance dictionary, ensuring we only include valid candidates
        dists = {idx: float(dist) for idx, dist in zip(cand_idxs, distances[0][:len(cand_idxs)])}

        # Invert distance to score: 1 / (1 + distance). Adding 1 avoids division by zero.
        return {
            idx: 1.0 / (1.0 + dist)
            for idx, dist in dists.items()
        }


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, index):
        self.index = index

    def get_scores(self,
                 query: str,
                 pool_size: int,
                 chunks: List[str]) -> Dict[int, float]:
        """
        Returns BM25 scores for top 'pool_size' keyed by global chunk index.
        """
        # Tokenize the query in the same way the index was built
        tokenized_query = preprocess_for_bm25(query)

        # Get scores for all documents in the corpus
        all_scores = self.index.get_scores(tokenized_query)

        # Find the indices of the top 'pool_size' scores
        num_candidates = min(pool_size, len(all_scores))
        top_k_indices = np.argpartition(-all_scores, kth=num_candidates-1)[:num_candidates]

        # Remove invalid indices and ensure they are within bounds
        top_k_indices = [i for i in top_k_indices if 0 <= i < len(chunks)]
        
        # Get the corresponding scores for the top indices
        top_scores = all_scores[top_k_indices]

        # Format the output as a dictionary of scores
        scores = {int(idx): float(score) for idx, score in zip(top_k_indices, top_scores)}

        return scores


class IndexKeywordRetriever(Retriever):
    name = "index_keywords"
    
    def __init__(self, extracted_index_path: os.PathLike, page_to_chunk_map_path: os.PathLike):
        """
        Retriever that uses textbook index keywords to boost chunks on relevant pages.
        
        Args:
            extracted_index_path: Path to extracted_index.json (keyword -> page numbers)
            page_to_chunk_map_path: Path to page_to_chunk_map.json (page -> chunk IDs)
        """
        import json
        nltk.download('wordnet', quiet=True)
        self.page_to_chunk_map = {}
        
        # Load and normalize index: lemmatize phrases as units
        # Build token->phrase mapping for fast lookup
        if os.path.exists(extracted_index_path):
            lemmatizer = WordNetLemmatizer()
            
            with open(extracted_index_path, 'r') as f:
                raw_index = json.load(f)
                self.phrase_to_pages = {}  # phrase -> pages
                self.token_to_phrases = {}  # token -> [phrases]
                
                for key, pages in raw_index.items():
                    # Lemmatize each word in the phrase but keep phrase together
                    key_lower = key.lower()
                    words = key_lower.split()
                    lemmatized_words = []
                    
                    for word in words:
                        cleaned = word.strip('.,!?()[]:"\'')
                        if not cleaned:
                            continue
                        lemmatized_words.append(self._lemmatize_word(cleaned, lemmatizer))
                    
                    lemmatized_phrase = ' '.join(lemmatized_words)
                    self.phrase_to_pages[lemmatized_phrase] = pages
                    
                    # Build reverse index: each token points to phrases containing it
                    for token in lemmatized_words:
                        if token not in self.token_to_phrases:
                            self.token_to_phrases[token] = []
                        self.token_to_phrases[token].append(lemmatized_phrase)
        else:
            self.phrase_to_pages = {}
            self.token_to_phrases = {}
        
        if os.path.exists(page_to_chunk_map_path):
            with open(page_to_chunk_map_path, 'r') as f:
                self.page_to_chunk_map = json.load(f)
    
    def get_scores(self, query: str, pool_size: int, chunks: List[str]) -> Dict[int, float]:
        """
        Returns scores for chunks that match index keywords.
        Score is proportional to the number of keyword hits.
        """
        keywords = self._extract_keywords(query)
        # chunk_id -> hit count
        chunk_hit_counts: Dict[int, int] = {} 
        
        # Match query keywords against index phrases (token overlap)
        for keyword in keywords:
            if keyword not in self.token_to_phrases:
                continue
            
            # Get all phrases containing this keyword token
            matching_phrases = self.token_to_phrases[keyword]
            
            for phrase in matching_phrases:
                page_numbers = self.phrase_to_pages[phrase]
                
                # Map pages to chunks
                for page_no in page_numbers:
                    chunk_ids = self.page_to_chunk_map.get(str(page_no), [])
                    for chunk_id in chunk_ids:
                        if chunk_id >= 0 and chunk_id < len(chunks):
                            chunk_hit_counts[chunk_id] = chunk_hit_counts.get(chunk_id, 0) + 1
        
        if not chunk_hit_counts:
            return {}
        
        # Normalize scores: more keyword hits = higher score
        max_hits = max(chunk_hit_counts.values())
        scores = {
            chunk_id: float(hit_count) / max_hits
            for chunk_id, hit_count in chunk_hit_counts.items()
        }
        
        return scores
    
    @staticmethod
    def _lemmatize_word(word: str, lemmatizer) -> str:
        """Lemmatize a word, trying noun then verb."""
        lemma = lemmatizer.lemmatize(word, pos='n')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, pos='v')
        return lemma
    
    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """Extract keywords from query by removing stopwords and lemmatizing."""
        
        stopwords = {
            "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in",
            "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", 
            "what", "how", "why", "when", "where", "who", "does", "do", "be"
        }
        
        lemmatizer = WordNetLemmatizer()
        words = query.lower().split()
        keywords = []
        for word in words:
            cleaned = word.strip('.,!?()[]:"\'')
            if not cleaned or cleaned in stopwords:
                continue
            keywords.append(IndexKeywordRetriever._lemmatize_word(cleaned, lemmatizer))
        return keywords