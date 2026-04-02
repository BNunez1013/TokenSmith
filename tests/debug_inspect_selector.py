from src.config import RAGConfig
from src.retriever import load_artifacts

INDEX_PREFIX = "textbook_index"

def main():
    cfg = RAGConfig()
    artifcats_dir = cfg.get_artifacts_directory()
    print("artifacts_dir=", artifcats_dir)
    faiss_idx, bm25_idx, chunks, sources, metadata = load_artifacts(artifacts_dir=artifcats_dir, index_prefix=INDEX_PREFIX)

    print(f"chunks={len(chunks)} metadata={len(metadata)}")

    sample_ids = [0, 1, 2, 10, 25, 50]
    for idx in sample_ids:
        if idx >= len(chunks) or idx >= len(metadata):
            continue
        meta = metadata[idx]
        print("=" * 80)
        print(f"idx: {idx}")
        print(f"meta.chunk_id: {meta.get('chunk_id')}")
        print(f"section: {meta.get('section')}")
        print(f"section_path: {meta.get('section_path')}")
        print(f"pages: {meta.get('page_numbers')}")
        print(f"preview: {meta.get('text_preview')}")
        print(f"chunk[:250]: {chunks[idx][:250]}")
if __name__ == "__main__":
    main()