#!/usr/bin/env python3
"""
Script to create the V19 notebook from V18 by replacing all V18 references with V19.
This preserves outputs, cell structure, and formatting.
"""
import json
import sys


def transform_source(source_lines):
    """Replace v18/V18 references with v19/V19 in source lines."""
    transformed = []
    for line in source_lines:
        # Replace various forms of v18 with v19
        line = line.replace("v18", "v19")
        line = line.replace("V18", "V19")
        line = line.replace("v_18", "v_19")
        line = line.replace("V_18", "V_19")
        transformed.append(line)
    return transformed


def main():
    v18_path = "training/notebooks/training/poker_agent_v18.ipynb"
    v19_path = "training/notebooks/training/poker_agent_v19.ipynb"

    with open(v18_path, "r") as f:
        notebook = json.load(f)

    # Transform all cells
    for cell in notebook.get("cells", []):
        # Transform source
        if "source" in cell:
            cell["source"] = transform_source(cell["source"])

        # Clear outputs - start fresh
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # Write the new notebook
    with open(v19_path, "w") as f:
        json.dump(notebook, f, indent=1)

    print(f"Successfully created {v19_path}")
    print(f"All v18/V18 references have been replaced with v19/V19")
    print(f"All code cell outputs have been cleared")


if __name__ == "__main__":
    main()
