import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np


def parse_log(log_path):
    print(f"Parsing log file: {log_path}")

    content = ""
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-16') as f:
                content = f.read()
        except UnicodeError:
            with open(log_path, 'r', encoding='utf-8') as f2:
                content = f2.read()
    else:
        print(f"Log file {log_path} does not exist.")
        return pd.DataFrame()

    chunks = re.split(r'={10,}\nRunning (\d+/6: .+?)\.\.\.\n={10,}', content)

    results = []

    for i in range(1, len(chunks), 2):
        exp_name_full = chunks[i].strip()
        exp_content = chunks[i+1]

        name_match = re.search(r'\d+/6: (.+) \(N=120\)', exp_name_full)
        if name_match:
            exp_id = name_match.group(1)
        else:
            exp_id = exp_name_full

        print(f"Processing experiment: {exp_id}")

        epoch_matches = re.findall(
            r'epoch (\d+): \[(.+?)\]', exp_content, re.DOTALL)

        if not epoch_matches:
            print(f"Warning: No epoch stats found for {exp_id}")
            continue

        final_epoch = epoch_matches[-1]
        epoch_num = int(final_epoch[0])
        stats_str = final_epoch[1]
        stats = [float(x.strip()) for x in stats_str.split(',')]

        res = {
            'Experiment': exp_id,
            'Epochs': epoch_num + 1,
            'Baseline': stats[0],
            'Best Sample': stats[1],
            'ACO@1': stats[2],
            'ACO@T': stats[3],
            'Diversity@1': stats[4],
            'Diversity@T': stats[5],
            'Bins': stats[6] if len(stats) > 6 else np.nan
        }
        results.append(res)

    return pd.DataFrame(results)


def generate_report(df, output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)

    if df.empty:
        print("No results to report.")
        return

    csv_path = os.path.join(output_dir, "results_table.csv")
    df.to_csv(csv_path, index=False)
    print("Saved table to", csv_path)

    plt.figure(figsize=(10, 6))
    plt.bar(df['Experiment'], df['ACO@T'], color='skyblue')
    plt.xlabel('Experiment Configuration')
    plt.ylabel('Best Fitness (Higher is Better)')
    plt.title('Comparison of GFACS Improvements (N=120) - Fitness')
    plt.axhline(y=0.87, color='r', linestyle='--',
                label='Baseline Fitness (~0.87)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "results_fitness.png")
    plt.savefig(chart_path)
    print("Saved fitness chart to", chart_path)

    if 'Bins' in df.columns and not df['Bins'].isna().all():
        plt.figure(figsize=(10, 6))
        plt.bar(df['Experiment'], df['Bins'], color='salmon')
        plt.xlabel('Experiment Configuration')
        plt.ylabel('Average Bins Used (Lower is Better)')
        plt.title('Comparison of GFACS Improvements (N=120) - Bins')
        plt.axhline(y=51.84, color='r', linestyle='--',
                    label='Paper Baseline (51.84)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_path_bins = os.path.join(output_dir, "results_bins.png")
        plt.savefig(chart_path_bins)
        print("Saved bins chart to", chart_path_bins)


if __name__ == "__main__":
    log_file = "final_results.log"
    if not os.path.exists(log_file):
        log_file = "../final_results.log"

    df = parse_log(log_file)
    print(df)
    generate_report(df)
