"""Upload fine-tuned model to Hugging Face Hub.

Usage:
    python scripts/upload_model_to_huggingface.py --repo-id your-username/model-name
    python scripts/upload_model_to_huggingface.py --repo-id your-username/model-name --folder-path ./path/to/model
    python scripts/upload_model_to_huggingface.py --repo-id your-username/model-name --private

Make sure to set HF_TOKEN environment variable or run `huggingface-cli login` first.
"""

import argparse

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a model folder to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--folder-path",
        type=str,
        default="./models/finetuned_embeddinggemma_books/merged_16bit",
        help="Path to the model folder to upload (default: ./models/finetuned_embeddinggemma_books/merged_16bit)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()

    # 1. Create the repo (skip if already exists)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        exist_ok=True,
        private=args.private,
    )

    # 2. Upload the entire folder
    api.upload_folder(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"Model uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
