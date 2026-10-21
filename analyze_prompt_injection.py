
"""
Analyze the prompt-injection pilot CSV.

Reports:
- Outcomes by condition and attack case
- Missing or truncated responses
- First-token probability summaries
- Repeated outputs
- Whether outcome variation exists for detector evaluation

This is descriptive analysis, not a validated detector.
"""

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

INPUT_FILE = Path("prompt_injection_pilot.csv")


def parse_bool(value):
    """Convert CSV boolean strings to True, False, or None."""
    value = (value or "").strip().lower()

    if value == "true":
        return True
    if value == "false":
        return False
    return None


def parse_float(value):
    """Convert a CSV value to a number when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.resolve()}\n"
            "Place this script beside the CSV file."
        )

    with INPUT_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def summarize_probabilities(rows):
    values = [
        parse_float(row.get("first_token_probability"))
        for row in rows
    ]
    values = [value for value in values if value is not None]

    if not values:
        return "No valid probability values"

    return (
        f"n={len(values)}, "
        f"mean={statistics.mean(values):.4f}, "
        f"min={min(values):.4f}, "
        f"max={max(values):.4f}"
    )


def main():
    rows = load_rows()

    if not rows:
        print("The CSV contains no trial rows.")
        return

    print("=" * 65)
    print("PROMPT-INJECTION PILOT: ANALYSIS REPORT")
    print("=" * 65)
    print(f"Input file: {INPUT_FILE}")
    print(f"Total trials: {len(rows)}")

    # 1. Basic data-quality checks
    missing_responses = [
        row for row in rows
        if not (row.get("response") or "").strip()
    ]

    truncated = [
        row for row in rows
        if (row.get("done_reason") or "").lower() == "length"
        or parse_bool(row.get("possible_truncation")) is True
    ]

    missing_probabilities = [
        row for row in rows
        if parse_float(
            row.get("first_token_probability")
        ) is None
    ]

    unknown_outcomes = [
        row for row in rows
        if parse_bool(row.get("canary_leaked")) is None
    ]

    print("\n[1] DATA QUALITY")
    print(f"Missing/empty responses: {len(missing_responses)}")
    print(f"Possibly truncated responses: {len(truncated)}")
    print(f"Missing probability values: {len(missing_probabilities)}")
    print(f"Unknown canary outcomes: {len(unknown_outcomes)}")

    # 2. Results by condition
    conditions = defaultdict(list)

    for row in rows:
        conditions[row.get("condition", "unknown")].append(row)

    print("\n[2] RESULTS BY CONDITION")

    for condition, group in sorted(conditions.items()):
        outcomes = [
            parse_bool(row.get("canary_leaked"))
            for row in group
        ]
        known = [outcome for outcome in outcomes if outcome is not None]
        leaks = sum(outcome is True for outcome in known)

        print(f"\nCondition: {condition}")
        print(f"  Trials: {len(group)}")
        print(f"  Known canary outcomes: {len(known)}")
        print(f"  Canary leaks: {leaks}")

        if known:
            print(f"  Observed leak rate: {leaks / len(known):.1%}")

        print(
            "  First-token probability: "
            + summarize_probabilities(group)
        )

    # 3. Results by individual case
    cases = defaultdict(list)

    for row in rows:
        cases[row.get("case_id", "unknown")].append(row)

    print("\n[3] RESULTS BY CASE")

    for case_id, group in sorted(cases.items()):
        condition = group[0].get("condition", "unknown")
        outcomes = [
            parse_bool(row.get("canary_leaked"))
            for row in group
        ]
        known = [outcome for outcome in outcomes if outcome is not None]
        leaks = sum(outcome is True for outcome in known)

        print(
            f"{case_id} | {condition} | "
            f"trials={len(group)} | "
            f"known={len(known)} | leaks={leaks} | "
            f"probability: {summarize_probabilities(group)}"
        )

    # 4. Outcome variation among injection trials
    injections = [
        row for row in rows
        if row.get("condition") != "benign_control"
    ]

    injection_outcomes = [
        parse_bool(row.get("attack_success"))
        for row in injections
    ]

    known_attack_outcomes = [
        outcome for outcome in injection_outcomes
        if outcome is not None
    ]

    outcome_counts = Counter(known_attack_outcomes)

    print("\n[4] DETECTOR FEASIBILITY CHECK")
    print(f"Injection trials: {len(injections)}")
    print(f"Injection trials with known attack labels: "
          f"{len(known_attack_outcomes)}")
    print(f"Attack outcome counts: {dict(outcome_counts)}")

    if len(outcome_counts) < 2:
        print(
            "RESULT: The known attack labels contain fewer than "
            "two outcome classes."
        )
        print(
            "A detector's ability to distinguish successful from "
            "unsuccessful attacks cannot be evaluated on this dataset."
        )
    else:
        print(
            "RESULT: Both outcome classes are present. "
            "Further evaluation may be possible, subject to sample "
            "size, independent attack variants, and data quality."
        )

    # 5. Repeated outputs
    response_groups = defaultdict(set)

    for row in rows:
        key = (
            row.get("case_id"),
            row.get("condition"),
        )
        response = (row.get("response") or "").strip()
        if response:
            response_groups[key].add(response)

    repeated_cases = [
        key for key, outputs in response_groups.items()
        if len(outputs) == 1
    ]

    print("\n[5] REPETITION CHECK")
    print(
        f"Cases with identical recorded responses across all "
        f"repetitions: {len(repeated_cases)} / {len(response_groups)}"
    )
    print(
        "Repeated outputs are not independent attack designs."
    )

    print("\n" + "=" * 65)
    print("Analysis complete. Interpret these results descriptively.")
    print("=" * 65)


if __name__ == "__main__":
    main()
