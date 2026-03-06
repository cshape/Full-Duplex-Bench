#!/usr/bin/env python3
"""
Aggregate latency metrics from all latency_intervals.json files in a directory.

Usage:
    python aggregate_latency.py --root_dir /path/to/dataset
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import statistics


def collect_latencies(root_dir: Path) -> Tuple[List[float], List[float]]:
    """
    Collect all latencies from latency_intervals.json files.
    
    Returns:
        (stop_latencies, response_latencies) where each is a list of durations in seconds
    """
    stop_latencies = []
    response_latencies = []
    
    # Find all latency_intervals.json files
    json_files = list(root_dir.rglob("latency_intervals.json"))
    
    if not json_files:
        print(f"⚠️  No latency_intervals.json files found in {root_dir}")
        return [], []
    
    print(f"Found {len(json_files)} latency files")
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Process stop latencies (overlaps)
            for interval in data.get("latency_stop_list", []):
                if len(interval) == 2:
                    duration = interval[1] - interval[0]
                    stop_latencies.append(duration)
            
            # Process response latencies (gaps)
            for interval in data.get("latency_resp_list", []):
                if len(interval) == 2:
                    duration = interval[1] - interval[0]
                    response_latencies.append(duration)
                    
        except Exception as e:
            print(f"⚠️  Error processing {json_file}: {e}")
    
    return stop_latencies, response_latencies


def print_stats(name: str, latencies: List[float]):
    """Print statistics for a list of latencies."""
    if not latencies:
        print(f"\n{name}: No data")
        return
    
    print(f"\n{name}:")
    print(f"  Samples:   {len(latencies)}")
    print(f"  Mean:      {statistics.mean(latencies):.3f}s")
    print(f"  Median:    {statistics.median(latencies):.3f}s")
    print(f"  Std Dev:   {statistics.stdev(latencies) if len(latencies) > 1 else 0:.3f}s")
    print(f"  Min:       {min(latencies):.3f}s")
    print(f"  Max:       {max(latencies):.3f}s")
    
    # Percentiles
    if len(latencies) >= 2:
        sorted_lat = sorted(latencies)
        p25 = sorted_lat[int(len(sorted_lat) * 0.25)]
        p75 = sorted_lat[int(len(sorted_lat) * 0.75)]
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        print(f"  P25:       {p25:.3f}s")
        print(f"  P75:       {p75:.3f}s")
        print(f"  P95:       {p95:.3f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate latency metrics from latency_intervals.json files"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing latency_intervals.json files"
    )
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        print(f"❌ Directory not found: {root_dir}")
        return
    
    print(f"📊 Aggregating latency metrics from: {root_dir}")
    print("=" * 60)
    
    stop_latencies, response_latencies = collect_latencies(root_dir)
    
    # Print results
    print_stats("📍 Stop Latency (overlap duration)", stop_latencies)
    print_stats("⏱️  Response Latency (turn-taking gap)", response_latencies)
    
    print("\n" + "=" * 60)
    print("✅ Done!\n")
    
    # Summary interpretation
    if response_latencies:
        avg = statistics.mean(response_latencies)
        print(f"💡 Average response time: {avg:.3f}s ({int(avg*1000)}ms)")
        if avg < 0.3:
            print("   → Excellent! Very low latency")
        elif avg < 0.8:
            print("   → Good latency for real-time conversation")
        elif avg < 1.5:
            print("   → Acceptable, but noticeable delay")
        else:
            print("   → High latency - may feel slow in conversation")


if __name__ == "__main__":
    main()

