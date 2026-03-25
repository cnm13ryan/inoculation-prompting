#!/usr/bin/env python3
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attribute_sweep_multi_seed_run import build_param_dir_name


def main() -> None:
    attrs_path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "ip_sweep"
        / "attributes_to_vary.json"
    )

    with open(attrs_path, "r", encoding="utf-8") as f:
        attribute_sets = json.load(f)

    for idx, attribute_set in enumerate(attribute_sets, start=1):
        print(f"{idx}: {build_param_dir_name(attribute_set)}")


if __name__ == "__main__":
    main()
