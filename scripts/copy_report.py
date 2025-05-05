#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path


def copy_file_with_check(source_file, destination_file):
    """
    Copies a single file from source to destination, checking existence first.

    Args:
        source_file (str): Path to the source file.
        destination_file (str): Path to the destination file.
    """
    source_path = Path(source_file)
    destination_path = Path(destination_file)

    print(f"Source file: {source_path}")
    print(f"Destination file: {destination_path}")

    # Check if the source file exists
    if not source_path.exists():
        print(f"Error: Source file '{source_path}' not found.", file=sys.stderr)
        sys.exit(1)
    if not source_path.is_file():
        print(f"Error: Source '{source_path}' is not a file.", file=sys.stderr)
        sys.exit(1)

    # Ensure the destination directory exists (optional, but good practice)
    dest_dir = destination_path.parent
    if dest_dir and not dest_dir.exists():
        try:
            print(f"Destination directory '{dest_dir}' does not exist. Creating...")
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(
                f"Error: Could not create destination directory '{dest_dir}': {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        shutil.copy2(source_path, destination_path)
        print(f"\nSuccessfully copied '{source_path}' to '{destination_path}'")
    except PermissionError:
        print(
            f"Error: Permission denied while trying to copy '{source_path}' to '{destination_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during copy: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy a source file to a destination file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source_file",
        help="Path to the source file to copy.",
        default=Path("p1/report/report.pdf"),
        nargs="?",
    )
    parser.add_argument(
        "destination_file",
        help="Path for the destination (copied) file.",
        default=Path("p1/Parent_ATCI_P1_report.pdf"),
        nargs="?",
    )

    args = parser.parse_args()

    copy_file_with_check(args.source_file, args.destination_file)

    print("\nScript finished.")
