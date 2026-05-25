"""One-click runner for validating the built-in `test sample` dataset."""

from __future__ import annotations

from pathlib import Path

from heatflux.tools.sample_validation import run_default_sample_validation


def main() -> None:
    root = Path(__file__).resolve().parent
    result = run_default_sample_validation(root)
    print(f"report_path={result.report_path}")
    print(f"generated_output={result.generated_output_path}")
    print(f"exact_match_first5={result.exact_match_first5}")


if __name__ == "__main__":
    main()
