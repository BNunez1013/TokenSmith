import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def save_bar_chart(labels, values, title, ylabel, output_path):
    plt.figure(figsize=(10,6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_grouped_bar_chart(benchmark_labels,
                           baseline_values,
                           constrained_values,
                           balanced_values,
                           title, ylabel, output_path,
                           baseline_label="baseline_pool20",
                           constrianed_label="max_chunks1_pool15", balanced_label="all_constraints_balanced"):
    
    x = np.arange(len(benchmark_labels))
    width = 0.25

    plt.figure(figsize=(12,6))
    plt.bar(x - width, baseline_values, width, label=baseline_label)
    plt.bar(x, constrained_values, width, label=constrianed_label)
    plt.bar(x + width, balanced_values, width, label=balanced_label)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, benchmark_labels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    results_path = Path("tests/results/metadata_retrieval_results.json")
    with results_path.open("r") as f:
        payload = json.load(f)
    
    summaries = payload["summary"]
    labels = [summary["label"] for summary in summaries] #Each test [___, ___, ___]
    section_coverages = [summary["avg_section_coverage"] for summary in summaries]
    ground_truths = [summary["avg_ground_truth_in_k"] for summary in summaries]

    np_section_coverages = np.array(section_coverages)
    np_ground_truths = np.array(ground_truths)

    save_bar_chart(labels, np_section_coverages[:, 0], "Average Section Coverage by Config",
                   "Average Section Coverage: Level 1", Path("tests/results/metadata_section_coverage_l1.png"))
    
    save_bar_chart(labels, np_ground_truths[:, 0], "Average Ground Truth in k by Config",
                   "Average Ground Truth in k: Level 1", Path("tests/results/metadata_ground_truth_l1.png"))
    
    save_bar_chart(labels, np_section_coverages[:, 1], "Average Section Coverage by Config",
                   "Average Section Coverage: Level 2", Path("tests/results/metadata_section_coverage_l2.png"))
    
    save_bar_chart(labels, np_ground_truths[:, 1], "Average Ground Truth in k by Config",
                   "Average Ground Truth in k: Level 2", Path("tests/results/metadata_ground_truth_l2.png"))
    
    save_bar_chart(labels, np_section_coverages[:, 2], "Average Section Coverage by Config",
                   "Average Section Coverage: Level 3", Path("tests/results/metadata_section_coverage_l3.png"))
    
    save_bar_chart(labels, np_ground_truths[:, 2], "Average Ground Truth in k by Config",
                   "Average Ground Truth in k: Level 3", Path("tests/results/metadata_ground_truth_l3.png"))
    
    baseline_summary = next(summary for summary in summaries if summary["label"] == "baseline_pool20")
    constrained_summary = next(summary for summary in summaries if summary["label"] == "max_chunks1_pool15")
    balanced_summary = next(summary for summary in summaries if summary["label"] == "all_constraints_balanced")

    baseline_map = {}
    for result in baseline_summary["results"]:
        for bench in result:
            baseline_map[bench["benchmark_id"]] = bench
    
    constrained_map = {}
    for result in constrained_summary["results"]:
        for bench in result:
            constrained_map[bench["benchmark_id"]] = bench
    
    all_constraint_map = {}
    for result in balanced_summary["results"]:
        for bench in result:
            all_constraint_map[bench["benchmark_id"]] = bench

    #baseline_map = {bench["benchmark_id"]: result for result in baseline_summary["results"]}
    #constrained_map = {result["benchmark_id"]: result for result in constrained_summary["results"]}

    benchmark_ids = list(baseline_map.keys())
    baseline_section = [baseline_map[benchmark_id]["section_coverage"] for benchmark_id in benchmark_ids]
    constrained_section = [constrained_map[benchmark_id]["section_coverage"] for benchmark_id in benchmark_ids]
    balanced_section = [all_constraint_map[benchmark_id]["section_coverage"] for benchmark_id in benchmark_ids]

    baseline_ground_truth = [baseline_map[benchmark_id]["ground_truth_in_k"] for benchmark_id in benchmark_ids]
    constrained_ground_truth = [constrained_map[benchmark_id]["ground_truth_in_k"] for benchmark_id in benchmark_ids]
    balanced_ground_truth = [all_constraint_map[benchmark_id]["ground_truth_in_k"] for benchmark_id in benchmark_ids]

    save_grouped_bar_chart(benchmark_ids, baseline_section, constrained_section, balanced_section, "Per-Benchmark Section Coverage",
                           "Section Coverage", "tests/results/metadata_per_benchmark_section_coverage.png")
    
    save_grouped_bar_chart(benchmark_ids, baseline_ground_truth, constrained_ground_truth, balanced_ground_truth, "Per-Benchmark Ground Truth in k",
                           "Ground Truth in k", "tests/results/metadata_per_benchmark_ground_truth.png")
    
    print(f"Saved plots to tests/results")
    
if __name__ == "__main__":
    main()

