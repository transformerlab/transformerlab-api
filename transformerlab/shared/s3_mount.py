"""
S3 bucket mounting utilities for user authentication.

This module handles mounting S3 buckets when users log in, similar to the functionality
in run.sh for multitenant mode.
"""

import os
import subprocess
from lab import WORKSPACE_DIR


def check_remote_path_exists_and_has_files(remote_path: str) -> bool:
    """
    Check if the BUCKET_REMOTE_PATH exists and contains any files.

    Args:
        remote_path: The path to check for existence and files

    Returns:
        True if the path exists and has files, False otherwise
    """
    try:
        # Check if path exists
        if not os.path.exists(remote_path):
            print(f"Remote path {remote_path} does not exist")
            return False

        # Check if it's a directory and has files
        if os.path.isdir(remote_path):
            # Check if directory has any files or subdirectories
            try:
                files = os.listdir(remote_path)
                has_content = len(files) > 0
                if not has_content:
                    print(f"Remote path {remote_path} exists but is empty")
                    return False
                else:
                    print(f"Remote path {remote_path} exists and has content")
                    return True
            except OSError:
                print(f"Remote path {remote_path} exists but cannot be read")
                return False
        else:
            # It's a file, so it exists
            print(f"Remote path {remote_path} exists as a file")
            return True

    except Exception as e:
        print(f"Error checking remote path {remote_path}: {e}")
        return False


def run_mountpoint_command(bucket_name: str, remote_workspace_dir: str, profile: str = "transformerlab-s3") -> bool:
    """
    Run the mount-s3 command to mount an S3 bucket.

    Args:
        bucket_name: Name of the S3 bucket to mount
        remote_workspace_dir: Local directory to mount the bucket to
        profile: AWS profile to use (default: transformerlab-s3)

    Returns:
        True if mount command succeeded, False otherwise
    """
    try:
        # Create the remote workspace directory if it doesn't exist
        os.makedirs(remote_workspace_dir, exist_ok=True)
        print(f"Created/verified remote workspace directory: {remote_workspace_dir}")

        # Run the mount-s3 command
        cmd = ["mount-s3", "--profile", profile, bucket_name, remote_workspace_dir]
        print(f"Running mount command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
        )

        if result.returncode == 0:
            print(f"Successfully mounted S3 bucket '{bucket_name}' to '{remote_workspace_dir}'")
            return True
        else:
            print(f"Mount command failed with return code {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("Mount command timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"Error running mount command: {e}")
        return False


def setup_user_s3_mount(user_id: str) -> bool:
    """
    Setup S3 bucket mounting for a user after login.

    This function checks if multitenant mode is enabled and if the remote path
    exists and has files. If not, it runs the mountpoint command.

    Args:
        user_id: The ID of the user who logged in

    Returns:
        True if setup was successful or not needed, False if there was an error
    """
    try:
        # Check if multitenant mode is enabled
        if os.getenv("TFL_MULTITENANT") != "true":
            print("Multitenant mode not enabled, skipping S3 mount setup")
            return True

        # Get required environment variables
        bucket_name = os.getenv("BUCKET_NAME")

        if not bucket_name:
            print("BUCKET_NAME not set in environment variables, skipping S3 mount")
            return True

        if not WORKSPACE_DIR:
            print("WORKSPACE_DIR not set in environment variables, skipping S3 mount")
            return True

        # Construct the remote path using WORKSPACE_DIR and BUCKET_REMOTE_PATH
        bucket_remote_path = os.path.join(WORKSPACE_DIR, "orgs", "org_2", "workspace")

        print(f"Setting up S3 mount for user {user_id}")
        print(f"Bucket: {bucket_name}, Remote path: {bucket_remote_path}")

        # Create the directory if it doesn't exist
        os.makedirs(bucket_remote_path, exist_ok=True)

        # Check if the remote path exists and has files
        if check_remote_path_exists_and_has_files(bucket_remote_path):
            print(f"Remote path {bucket_remote_path} already exists with content, skipping mount")
            return True

        # Run the mountpoint command
        success = run_mountpoint_command(bucket_name, bucket_remote_path)

        if success:
            print(f"S3 mount setup completed successfully for user {user_id}")
        else:
            print(f"S3 mount setup failed for user {user_id}")

        return success

    except Exception as e:
        print(f"Error setting up S3 mount for user {user_id}: {e}")
        return False
