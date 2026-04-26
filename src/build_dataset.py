"""
Build complete dataset by extracting attention patterns for all prompts.
"""

import json
import os

import pandas as pd
from tqdm import tqdm

from attention_extraction import AttentionExtractor


def build_dataset(
    prompts_file: str = "data/raw/task_prompts.json",
    output_csv: str = "data/processed/attention_dataset.csv",
    output_json: str = "data/processed/attention_dataset.json",
):
    """
    Build complete dataset with attention features.

    Args:
        prompts_file: Path to JSON file with task prompts
        output_csv: Path to save CSV
        output_json: Path to save JSON
    """
    # Load prompts
    with open(prompts_file, "r") as f:
        task_prompts = json.load(f)

    # Initialize extractor
    extractor = AttentionExtractor()

    # Collect data
    dataset = []

    print("Extracting attention patterns...")
    for task_type, prompts in task_prompts.items():
        print(f"\nProcessing {task_type}...")

        for prompt in tqdm(prompts, desc=task_type):
            try:
                # Extract attention
                attn_patterns, tokens = extractor.get_attention_patterns(prompt)

                # Extract features
                features = extractor.extract_features(attn_patterns)

                # Store
                entry = {
                    "task_type": task_type,
                    "prompt": prompt,
                    "n_tokens": len(tokens),
                    **features,  # Unpack all attention features
                }

                dataset.append(entry)

            except Exception as e:
                print(f"\nError on prompt '{prompt}': {e}")
                continue

    # Convert to DataFrame
    df = pd.DataFrame(dataset)

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2)

    # Print statistics
    print(f"\n{'=' * 60}")
    print("Dataset Statistics:")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(df)}")
    print("\nSamples per task:")
    print(df["task_type"].value_counts())
    print(
        f"\nFeatures extracted: {len([c for c in df.columns if c.startswith(('entropy', 'induction', 'self_attn', 'first_tok', 'spread', 'max_attn'))])}"
    )
    print("\nDataset saved to:")
    print(f"  - {output_csv}")
    print(f"  - {output_json}")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    df = build_dataset()
