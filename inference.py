"""
inference.py
============
Entry-point for the Email Triage RL Simulation.

Run
---
    python inference.py                  # runs all three difficulty levels
    python inference.py --task easy      # runs a single level
    python inference.py --episodes 5     # repeats each level N times
"""

import argparse

from env import execute_task, Q_MEMORY


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Email Triage Reinforcement Learning Simulation"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty level to run (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes (training passes) per task (default: 1)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    print("\n" + "=" * 55)
    print("   Email Triage RL Simulation  —  START")
    print("=" * 55)
    print(f"   Tasks    : {tasks}")
    print(f"   Episodes : {args.episodes}")
    print(f"   Q-table  : starts empty, grows during run")

    results: dict[str, float] = {}

    for level in tasks:
        level_total = 0.0
        for episode in range(1, args.episodes + 1):
            if args.episodes > 1:
                print(f"\n--- Episode {episode}/{args.episodes} ---")
            score = execute_task(level)
            level_total += score
        results[level] = round(level_total / args.episodes, 4)

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 55)
    print("  FINAL AVERAGE SCORES")
    print("=" * 55)
    for level, avg in results.items():
        print(f"  {level:<8}: {avg}")

    print(f"\n  Q-table entries learned : {len(Q_MEMORY)}")
    print("\n  Top Q-table entries (state, action) → value:")
    top = sorted(Q_MEMORY.items(), key=lambda x: x[1], reverse=True)[:10]
    for (state, action), val in top:
        print(f"    ('{state}', action={action}) → {val:.4f}")

    print("\n" + "=" * 55)
    print("  DONE")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
