import os
import re
import csv
import subprocess
import argparse
from pathlib import Path

DEFAULT_RUN_NAME    = "my_experiment"
DEFAULT_SEEDS       = [1005, 89, 42]
DEFAULT_CONFIG_PATH = "../configs/adaround/FYP_CONFIG_FILE.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default=DEFAULT_RUN_NAME)
    parser.add_argument('--config',   default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--seeds',    nargs='+', type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()

def read_yaml_raw(path):
    with open(path, 'r') as f:
        return f.read()

def write_yaml_raw(path, content):
    with open(path, 'w') as f:
        f.write(content)

def set_seed_in_yaml(content, seed):
    """Replace seed value in yaml content using regex, preserving formatting."""
    return re.sub(r'(seed:\s*)\d+', rf'\g<1>{seed}', content)

def parse_accuracy_from_file(log_path: Path):
    """
    Parse Top-1 and Top-5 accuracy from log file.
    Looks for lines like:  * Acc@1 71.234 Acc@5 90.123
    """
    top1, top5 = None, None
    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(r'Acc@1\s+([\d.]+).*Acc@5\s+([\d.]+)', line)
            if m:
                top1 = float(m.group(1))
                top5 = float(m.group(2))
    return top1, top5

def main():
    args = parse_args()
    RUN_NAME    = args.run_name
    CONFIG_PATH = args.config
    SEEDS       = args.seeds

    run_dir = Path.home() / "FYPRUN_DATA" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    csv_path = run_dir / f"{RUN_NAME}.csv"
    config_path = Path(CONFIG_PATH)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Read original config once
    original_yaml = read_yaml_raw(config_path)

    results = []  # list of (row_name, top1, top5)

    for seed in SEEDS:
        row_name = f"{RUN_NAME}_{seed}"
        print(f"\n{'='*60}")
        print(f"  Running: {row_name}")
        print(f"{'='*60}")

        # Overwrite seed in config
        modified_yaml = set_seed_in_yaml(original_yaml, seed)
        write_yaml_raw(config_path, modified_yaml)

        # Set env vars so ALL smoothquant JSON outputs go to our run dir
        env = os.environ.copy()
        env['SQ_STATS_PATH']             = str(run_dir / f"smoothquant_stats_{row_name}.json")
        env['POST_QUANT_STATS_PATH']     = str(run_dir / f"post_quant_stats_{row_name}.json")
        env['BASELINE_QUANT_STATS_PATH'] = str(run_dir / f"baseline_quant_stats_{row_name}.json")

        # Run verify_ptq.py, streaming stdout+stderr to log file
        cmd = ['python3', 'verify_ptq.py', '--config', str(config_path)]
        print(f"  CMD: {' '.join(cmd)}")

        log_path = run_dir / f"{row_name}.log"
        with open(log_path, 'w') as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=log_file,
                text=True,
                env=env
            )

        if result.returncode != 0:
            print(f"  WARNING Process exited with code {result.returncode}")

        # Parse accuracy from log file
        top1, top5 = parse_accuracy_from_file(log_path)

        if top1 is None:
            print(f"  WARNING Could not parse accuracy from output for seed {seed}")
            top1, top5 = 'ERROR', 'ERROR'

        print(f"  → Top1: {top1}, Top5: {top5}")
        results.append((row_name, top1, top5))

    # Delete all log files now that we're done
    # for seed in SEEDS:
    #     log_path = run_dir / f"{RUN_NAME}_{seed}.log"
    #     if log_path.exists():
    #         log_path.unlink()

    # Restore original yaml seed
    write_yaml_raw(config_path, original_yaml)
    print(f"\nRestored original config seed.")

    # Compute averages
    numeric = [(r, t1, t5) for r, t1, t5 in results
               if isinstance(t1, float) and isinstance(t5, float)]
    if numeric:
        avg_top1 = sum(t1 for _, t1, _ in numeric) / len(numeric)
        avg_top5 = sum(t5 for _, _, t5 in numeric) / len(numeric)
    else:
        avg_top1, avg_top5 = 'N/A', 'N/A'

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'Top1', 'Top5'])
        for row in results:
            writer.writerow(row)
        writer.writerow([f"{RUN_NAME}_AVERAGE", avg_top1, avg_top5])

    print(f"\n{'='*60}")
    print(f"  DONE. Results saved to: {csv_path}")
    print(f"{'='*60}")
    print(f"  Average → Top1: {avg_top1}, Top5: {avg_top5}")

if __name__ == '__main__':
    main()
