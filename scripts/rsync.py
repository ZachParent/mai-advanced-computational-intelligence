#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def sync_directory_rsync(
    remote_host, local_dir, remote_dir, direction, exclude_patterns=None
):
    """
    Synchronizes a directory between local and remote using rsync.

    Args:
        remote_host (str): SSH host alias from your config (e.g., 'my_server').
        local_dir (str): Path to the local directory.
        remote_dir (str): Path to the remote directory.
        direction (str): 'send' (local -> remote) or 'receive' (remote -> local).
        exclude_patterns (list, optional): List of patterns to exclude. Defaults to None.
    """
    # Ensure local directory exists for 'send'
    if direction == "send" and not os.path.isdir(local_dir):
        print(f"Error: Local directory '{local_dir}' not found.")
        sys.exit(1)

    # Use rsync instead of scp
    # -a: archive mode (recursive, preserves permissions, times, groups, etc.)
    # -v: verbose (shows files being transferred)
    # -z: compress file data during transfer
    # -e ssh: specify ssh as the remote shell (often default, but good practice)
    # --delete: (optional) uncomment if you want files deleted on the destination
    #           if they no longer exist on the source
    rsync_command = ["rsync", "-avz", "-e", "ssh", "--delete"]

    # Add exclude patterns if provided
    if exclude_patterns:
        for pattern in exclude_patterns:
            rsync_command.append(f"--exclude={pattern}")

    # Ensure trailing slashes for rsync to copy contents correctly
    local_path_rsync = os.path.join(local_dir, "")  # Adds trailing slash if needed
    remote_path_rsync = f"{remote_host}:{os.path.join(remote_dir, '')}"

    try:
        if direction == "send":
            source = local_path_rsync
            destination = remote_path_rsync
            print(f"Sending '{source}' to '{destination}' (excluding patterns)...")
            rsync_command.extend([source, destination])

        elif direction == "receive":
            source = remote_path_rsync
            destination = local_path_rsync
            print(
                f"Receiving from '{source}' to '{destination}' (excluding patterns)..."
            )
            rsync_command.extend([source, destination])
            # Ensure local directory exists before copying into it
            os.makedirs(destination, exist_ok=True)

        else:
            print(f"Error: Invalid direction '{direction}'. Use 'send' or 'receive'.")
            sys.exit(1)

        print(f"Executing: {' '.join(rsync_command)}")  # Show the command being run
        # Using Popen to stream output might be better for long transfers,
        # but run is simpler for now.
        process = subprocess.run(
            rsync_command, check=True, capture_output=True, text=True
        )
        print("Sync successful!")
        if process.stdout:  # Show rsync output if any
            print("--- RSYNC STDOUT ---")
            print(process.stdout)
        if process.stderr:  # Show rsync errors/warnings if any
            print("--- RSYNC STDERR ---")
            print(process.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error during rsync operation (Exit code: {e.returncode}):")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        # Updated error message for rsync
        print("Error: 'rsync' command not found. Is rsync installed and in your PATH?")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronize a directory with a remote host using rsync and SSH config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "direction",
        choices=["send", "receive"],
        help="Direction of synchronization: 'send' (local->remote) or 'receive' (remote->local).",
    )
    parser.add_argument(
        "--remote_host",
        help="SSH host alias defined in your ~/.ssh/config file.",
        default="atci-sweden",
    )
    parser.add_argument(
        "--local_dir",
        help="Path to the local directory.",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--remote_dir",
        help="Path to the remote directory.",
        default="/workspace/atci",
    )

    # Add argument for excludes
    parser.add_argument(
        "--exclude",
        action="append",  # Allows specifying --exclude multiple times
        default=[],  # Default to an empty list if not provided
        help="Pattern to exclude (e.g., '*venv/', '*.pyc'). Can be used multiple times.",
    )

    args = parser.parse_args()

    # Normalize paths (optional, but good practice)
    local_path = os.path.abspath(os.path.expanduser(args.local_dir))
    # No normalization for remote path as it's interpreted by the remote server

    # Add the specific exclude pattern you requested if not already passed via CLI
    exclude_list = args.exclude
    if "*venv/" not in exclude_list:  # Add your default exclude if not specified
        exclude_list.extend(
            ["*venv/", "*.pyc", "*.pyo", ".vscode/", ".DS_Store", ".git/"]
        )

    sync_directory_rsync(
        args.remote_host, local_path, args.remote_dir, args.direction, exclude_list
    )
