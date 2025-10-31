#!/usr/bin/env python3
"""
Script to create a zip archive of the apps folder for easy download.
Usage: python zip_apps_folder.py [output_path]
"""
import argparse
import sys
import zipfile
from pathlib import Path
from datetime import datetime


def zip_apps_folder(output_path: str | None = None) -> Path:
    """
    Create a zip archive of the apps folder.
    
    Args:
        output_path: Optional custom path for the output zip file.
                    If not provided, creates 'apps_YYYYMMDD_HHMMSS.zip' in current directory.
    
    Returns:
        Path to the created zip file.
    """
    # Get the project root (parent of tools directory)
    project_root = Path(__file__).resolve().parent.parent
    apps_dir = project_root / 'apps'
    
    if not apps_dir.exists():
        raise FileNotFoundError(f"Apps directory not found at {apps_dir}")
    
    # Determine output path
    if output_path:
        zip_path = Path(output_path)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_path = Path.cwd() / f'apps_{timestamp}.zip'
    
    # Create zip file
    print(f"Creating zip archive of apps folder...")
    print(f"Source: {apps_dir}")
    print(f"Output: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files in apps directory
        for file_path in apps_dir.rglob('*'):
            if file_path.is_file():
                # Calculate the archive name (relative path from project root)
                arcname = file_path.relative_to(project_root)
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")
    
    print(f"\nZip archive created successfully: {zip_path}")
    print(f"Size: {zip_path.stat().st_size / 1024:.2f} KB")
    
    return zip_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Create a zip archive of the apps folder for download',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create zip with default name (apps_YYYYMMDD_HHMMSS.zip)
  python zip_apps_folder.py
  
  # Create zip with custom name
  python zip_apps_folder.py my_apps.zip
  
  # Create zip in specific directory
  python zip_apps_folder.py /path/to/output/apps.zip
        """
    )
    parser.add_argument(
        'output',
        nargs='?',
        help='Output path for the zip file (optional)'
    )
    
    args = parser.parse_args()
    
    try:
        zip_path = zip_apps_folder(args.output)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
