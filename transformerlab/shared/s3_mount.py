"""
S3 bucket mounting utilities for user authentication.

This module handles mounting S3 buckets when users log in, similar to the functionality
in run.sh for multitenant mode. Supports both mount-s3 (Linux) and s3fs (macOS).
"""

import os
import platform
import subprocess


def is_macos() -> bool:
    """Check if the current platform is macOS."""
    return platform.system() == "Darwin"


def get_s3_mount_command() -> str:
    """
    Get the appropriate S3 mounting command based on the platform.
    
    Returns:
        The command to use for S3 mounting ('mount-s3' for Linux, 's3fs' for macOS)
    """
    if is_macos():
        return "s3fs"
    else:
        return "mount-s3"


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


def run_s3_mount_command(bucket_name: str, remote_workspace_dir: str, profile: str = "transformerlab-s3") -> bool:
    """
    Run the appropriate S3 mounting command (mount-s3 for Linux, s3fs for macOS).

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

        mount_command = get_s3_mount_command()
        
        if mount_command == "s3fs":
            # s3fs command for macOS
            # s3fs supports AWS profiles through ~/.aws/credentials
            # Format: s3fs bucket_name mount_point -o profile=profile_name
            cmd = ["s3fs", bucket_name, remote_workspace_dir, "-o", f"profile={profile}"]
        else:
            # mount-s3 command for Linux
            cmd = ["mount-s3", "--profile", profile, bucket_name, remote_workspace_dir]
        
        print(f"Running mount command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
        )

        if result.returncode == 0:
            print(f"Successfully mounted S3 bucket '{bucket_name}' to '{remote_workspace_dir}' using {mount_command}")
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


def setup_user_s3_mount(user_id: str, organization_id: str | None = None) -> bool:
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
        HOME_DIR = os.path.join(os.path.expanduser("~"), ".transformerlab")

        if not bucket_name:
            print("BUCKET_NAME not set in environment variables, skipping S3 mount")
            return True

        if not os.path.exists(HOME_DIR):
            print(f"{HOME_DIR} does not exist, skipping S3 mount")
            return True

        # Construct the remote path using HOME_DIR and organization_id (fallback to org_2)
        org_dir = organization_id or "org_2"
        bucket_remote_path = os.path.join(HOME_DIR, "orgs", org_dir, "workspace")

        print(f"Setting up S3 mount for user {user_id}")
        print(f"Organization: {org_dir}")
        print(f"Bucket: {bucket_name}, Remote path: {bucket_remote_path}")

        # Create the directory if it doesn't exist
        os.makedirs(bucket_remote_path, exist_ok=True)

        # Check if the remote path exists and has files
        if check_remote_path_exists_and_has_files(bucket_remote_path):
            print(f"Remote path {bucket_remote_path} already exists with content, skipping mount")
            return True

        # Run the S3 mount command (mount-s3 for Linux, s3fs for macOS)
        success = run_s3_mount_command(bucket_name, bucket_remote_path)

        if success:
            print(f"S3 mount setup completed successfully for user {user_id}")
        else:
            print(f"S3 mount setup failed for user {user_id}")

        return success

    except Exception as e:
        print(f"Error setting up S3 mount for user {user_id}: {e}")
        return False
