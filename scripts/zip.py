#!/usr/bin/env python3
import argparse
import fnmatch
import os
import sys
import zipfile

# Hardcoded list of patterns to exclude
# Matches files/dirs anywhere in the tree
EXCLUDE_PATTERNS = [
    ".git*",  # Git directory and related files
    ".vscode/",  # VSCode settings
    "__pycache__/",  # Python cache
    "*.pyc",  # Python compiled files
    "*.pyo",  # Python optimized files
    "*venv/",  # Python virtual environments (common names)
    ".DS_Store",  # macOS metadata
    "*.zip",  # Don't include zip files (avoids self-inclusion)
    "data/02_videos/",  # Data directory
    "data/03_agents/",  # Data directory
    "report/",  # Report directory
]


def should_exclude(path_relative_to_source, is_dir):
    """Check if a given path relative to the source dir should be excluded."""
    # Normalize path separators for matching
    path_parts = path_relative_to_source.split(os.sep)
    basename = path_parts[-1]

    for pattern in EXCLUDE_PATTERNS:
        # Check direct match on basename (e.g., '*.pyc', '.DS_Store')
        if fnmatch.fnmatchcase(
            basename, pattern.strip("/")
        ):  # Use fnmatchcase for consistency
            return True
        # Check if pattern targets a directory (e.g., '.git/', '*venv/')
        if pattern.endswith("/"):
            # Check if any directory component matches the pattern
            # Add '/' suffix for directory matching comparison
            normalized_pattern = pattern.strip("/") + "/"
            current_path_normalized = ""
            for part in path_parts:
                current_path_normalized += part + "/"
                if fnmatch.fnmatchcase(current_path_normalized, normalized_pattern):
                    return True
                # Also match if the pattern matches a directory part directly
                if fnmatch.fnmatchcase(part + "/", normalized_pattern):
                    return True
            # Special check for top-level dir patterns if path is a dir itself
            if is_dir and fnmatch.fnmatchcase(basename + "/", pattern):
                return True

    return False


def create_zip_archive(source_dir, zip_basename):
    """
    Creates a zip archive from a source directory, excluding specified patterns.

    Args:
        source_dir (str): Path to the source directory to zip.
        zip_basename (str): The desired base name for the output zip file
                             and the root directory within the zip file.
    """
    source_path = os.path.abspath(os.path.expanduser(source_dir))
    zip_filename = f"{zip_basename}.zip"

    if not os.path.isdir(source_path):
        print(f"Error: Source directory '{source_path}' not found.")
        sys.exit(1)

    print(f"Source directory: {source_path}")
    print(f"Output zip file: {zip_filename}")
    print(f"Excluding patterns: {EXCLUDE_PATTERNS}")

    try:
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            print(f"Creating archive '{zip_filename}'...")
            items_added_count = 0
            items_excluded_count = 0

            for root, dirs, files in os.walk(source_path, topdown=True):
                # --- Exclude directories ---
                original_dirs = list(dirs)  # Copy before modifying
                dirs[:] = [
                    d
                    for d in original_dirs
                    if not should_exclude(
                        os.path.relpath(os.path.join(root, d), source_path), True
                    )
                ]
                excluded_dirs_count_in_iter = len(original_dirs) - len(dirs)
                items_excluded_count += excluded_dirs_count_in_iter
                if excluded_dirs_count_in_iter > 0:
                    print(
                        f"  (Excluded {excluded_dirs_count_in_iter} subdirectories in {os.path.relpath(root, source_path) or '.'})"
                    )

                # --- Process files ---
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, source_path)

                    if should_exclude(relative_path, False):
                        # print(f"  Excluding file: {relative_path}") # Optional: Verbose logging
                        items_excluded_count += 1
                        continue

                    # Path inside the zip file: root directory + relative path
                    archive_path = os.path.join(zip_basename, relative_path)

                    # print(f"  Adding file: {relative_path} as {archive_path}") # Optional: Verbose logging
                    zipf.write(full_path, arcname=archive_path)
                    items_added_count += 1

            print(f"Archive creation successful!")
            print(f" - Items added: {items_added_count}")
            print(f" - Items excluded: {items_excluded_count}")
            print(f"Archive saved as: {os.path.abspath(zip_filename)}")

    except FileNotFoundError:
        # Should be caught by the initial check, but good practice
        print(f"Error: Source directory '{source_path}' disappeared during zipping.")
        sys.exit(1)
    except PermissionError:
        print(
            f"Error: Permission denied while accessing files in '{source_path}' or writing to '{zip_filename}'."
        )
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during zipping: {e}")
        try:  # Attempt to remove partially created zip file on error
            if os.path.exists(zip_filename):
                os.remove(zip_filename)
                print(f"Removed partially created archive '{zip_filename}'.")
        except Exception as cleanup_e:
            print(f"Additionally, failed to remove partial archive: {cleanup_e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a zip archive of a directory, excluding specified patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source_dir",
        help="Path to the directory to zip.",
        default="p1",
    )
    parser.add_argument(
        "--zip_name",
        help="Base name for the output zip file ('.zip' will be appended) and the root directory inside the archive.",
        default="Parent_ATCI_P1",
    )
    # Future enhancement: Could add an --exclude argument like rsync.py
    # parser.add_argument(
    #     "--exclude",
    #     action="append",
    #     help="Additional pattern to exclude. Can be used multiple times.",
    # )

    args = parser.parse_args()

    # Basic validation for zip_name (avoid empty or path-like names)
    if not args.zip_name or "/" in args.zip_name or "\\" in args.zip_name:
        print(
            f"Error: Invalid zip_name '{args.zip_name}'. It cannot be empty or contain path separators."
        )
        sys.exit(1)

    # Normalize source path before passing to function
    source_directory = os.path.abspath(os.path.expanduser(args.source_dir))

    create_zip_archive(source_directory, args.zip_name)

    print("Script finished.")
