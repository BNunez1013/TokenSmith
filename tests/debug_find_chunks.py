import argparse
from pathlib import Path

from src.config import RAGConfig
from src.retriever import load_artifacts


INDEX_PREFIX = "textbook_index"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect indexed chunks by section path, keyword, or chunk id."
    )
    parser.add_argument(
        "--section",
        help="Case-insensitive substring match on metadata section_path.",
    )
    parser.add_argument(
        "--keyword",
        help="Case-insensitive substring match on chunk text or text preview.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        help="Explicit chunk ids to inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of matches to print for section/keyword search.",
    )
    parser.add_argument(
        "--show-text",
        type=int,
        default=250,
        help="Number of chunk characters to print.",
    )
    return parser


def _print_chunk(idx: int, chunks: list[str], metadata: list[dict], show_text: int) -> None:
    if idx < 0 or idx >= len(chunks) or idx >= len(metadata):
        print(f"[skip] invalid chunk id: {idx}")
        return

    meta = metadata[idx]
    print("=" * 80)
    print(f"idx: {idx}")
    print(f"meta.chunk_id: {meta.get('chunk_id')}")
    print(f"section: {meta.get('section')}")
    print(f"section_path: {meta.get('section_path')}")
    print(f"pages: {meta.get('page_numbers')}")
    print(f"preview: {meta.get('text_preview')}")
    print(f"chunk[:{show_text}]: {chunks[idx][:show_text]}")


def _match_section(metadata: list[dict], query: str) -> list[int]:
    query = query.lower()
    matches = []
    for idx, meta in enumerate(metadata):
        section_path = str(meta.get("section_path", "")).lower()
        if query in section_path:
            matches.append(idx)
    return matches


def _match_keyword(chunks: list[str], metadata: list[dict], query: str) -> list[int]:
    query = query.lower()
    matches = []
    for idx, chunk in enumerate(chunks):
        preview = str(metadata[idx].get("text_preview", ""))
        haystack = f"{chunk}\n{preview}".lower()
        if query in haystack:
            matches.append(idx)
    return matches


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = RAGConfig()
    artifacts_dir = cfg.get_artifacts_directory()

    _, _, chunks, _, metadata = load_artifacts(
        artifacts_dir=artifacts_dir,
        index_prefix=INDEX_PREFIX,
    )

    print(f"artifacts_dir={artifacts_dir}")
    print(f"chunks={len(chunks)} metadata={len(metadata)}")

    seen = set()

    if args.ids:
        for idx in args.ids:
            seen.add(idx)
            _print_chunk(idx, chunks, metadata, args.show_text)

    if args.section:
        matches = _match_section(metadata, args.section)
        print(f"\n[section] '{args.section}' -> {len(matches)} matches")
        for idx in matches[: args.limit]:
            if idx in seen:
                continue
            seen.add(idx)
            _print_chunk(idx, chunks, metadata, args.show_text)

    if args.keyword:
        matches = _match_keyword(chunks, metadata, args.keyword)
        print(f"\n[keyword] '{args.keyword}' -> {len(matches)} matches")
        for idx in matches[: args.limit]:
            if idx in seen:
                continue
            seen.add(idx)
            _print_chunk(idx, chunks, metadata, args.show_text)

    if not any([args.ids, args.section, args.keyword]):
        print("\nPass one of: --ids, --section, --keyword")


if __name__ == "__main__":
    main()
