#!/usr/bin/env python3
"""
PyPI Requirements Version Fixer

This script reads a requirements.txt file and pins dynamic packages to their
latest version as of a specified date using the PyPI API.
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from typing import Optional, Tuple
import requests
from packaging.version import parse, Version
from packaging.specifiers import SpecifierSet


def parse_requirement_line(line: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Parse a requirement line to extract package name, version spec, and extras.
    
    Returns:
        Tuple of (package_name, version_spec, original_line)
    """
    # Skip empty lines and comments
    line = line.strip()
    if not line or line.startswith('#'):
        return (None, None, line)
    
    # Handle -r and -c directives
    if line.startswith('-r ') or line.startswith('--requirement '):
        return (None, None, line)
    if line.startswith('-c ') or line.startswith('--constraint '):
        return (None, None, line)
    
    # Handle other pip options
    if line.startswith('-'):
        return (None, None, line)
    
    # Regular expression to parse package lines
    # Matches: package-name[extras]>=version,<version ; markers
    pattern = r'^([a-zA-Z0-9._-]+)(\[[^\]]*\])?([^;#]*)?([;#].*)?$'
    match = re.match(pattern, line)
    
    if not match:
        return (None, None, line)
    
    package_name = match.group(1)
    extras = match.group(2) or ''
    version_spec = match.group(3).strip() if match.group(3) else ''
    markers = match.group(4) or ''
    
    # Check if version is already specified
    if version_spec and any(op in version_spec for op in ['==', '>=', '<=', '>', '<', '~=', '!=']):
        # Version already specified, skip
        return (None, None, line)
    
    return (package_name, None, line)


def get_latest_version_before_date(package_name: str, target_date: datetime) -> Optional[str]:
    """
    Query PyPI API to find the latest version of a package released before the target date.
    
    Args:
        package_name: Name of the PyPI package
        target_date: Date to find the latest version before
        
    Returns:
        Version string or None if not found
    """
    try:
        # Query PyPI JSON API
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
        if response.status_code != 200:
            print(f"Warning: Could not fetch data for {package_name}: HTTP {response.status_code}", file=sys.stderr)
            return None
        
        data = response.json()
        releases = data.get('releases', {})
        
        valid_versions = []
        
        for version_str, release_info in releases.items():
            if not release_info:
                continue
            
            # Skip pre-releases and dev versions unless no stable versions exist
            try:
                version = parse(version_str)
                # Get the upload time of the first file in this release
                upload_time_str = release_info[0].get('upload_time_iso_8601')
                if not upload_time_str:
                    upload_time_str = release_info[0].get('upload_time')
                    if upload_time_str:
                        # Convert from older format: "2023-06-15T10:30:45"
                        upload_time = datetime.fromisoformat(upload_time_str.replace('Z', '+00:00'))
                    else:
                        continue
                else:
                    upload_time = datetime.fromisoformat(upload_time_str.replace('Z', '+00:00'))
                
                # Ensure timezone awareness
                if upload_time.tzinfo is None:
                    upload_time = upload_time.replace(tzinfo=timezone.utc)
                if target_date.tzinfo is None:
                    target_date = target_date.replace(tzinfo=timezone.utc)
                
                if upload_time <= target_date:
                    valid_versions.append((version, version_str, upload_time))
            except Exception as e:
                print(f"Warning: Could not parse version {version_str} for {package_name}: {e}", file=sys.stderr)
                continue
        
        if not valid_versions:
            print(f"Warning: No versions found for {package_name} before {target_date}", file=sys.stderr)
            return None
        
        # First try to get only stable versions
        stable_versions = [(v, vs, ut) for v, vs, ut in valid_versions if not v.is_prerelease and not v.is_devrelease]
        
        if stable_versions:
            # Sort by version and get the latest
            stable_versions.sort(key=lambda x: x[0])
            return stable_versions[-1][1]
        else:
            # If no stable versions, use the latest pre-release
            valid_versions.sort(key=lambda x: x[0])
            return valid_versions[-1][1]
            
    except requests.RequestException as e:
        print(f"Error: Network error while fetching {package_name}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: Unexpected error while processing {package_name}: {e}", file=sys.stderr)
        return None


def process_requirements_file(input_file: str, output_file: str, target_date: datetime):
    """
    Process a requirements file and pin dynamic packages to specific versions.
    
    Args:
        input_file: Path to input requirements.txt
        output_file: Path to output requirements.txt (can be same as input)
        target_date: Date to find versions for
    """
    updated_lines = []
    changes_made = False
    
    print(f"Processing requirements file: {input_file}")
    print(f"Target date: {target_date.strftime('%Y-%m-%d')}")
    print("-" * 50)
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {input_file} not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {input_file}: {e}", file=sys.stderr)
        sys.exit(1)
    
    for line_num, line in enumerate(lines, 1):
        original_line = line.rstrip('\n')
        package_name, version_spec, parsed_line = parse_requirement_line(original_line)
        
        if package_name and version_spec is None:
            # This is a dynamic package, need to pin it
            print(f"Processing {package_name}... ", end='', flush=True)
            
            latest_version = get_latest_version_before_date(package_name, target_date)
            
            if latest_version:
                # Check for extras and markers
                extras_pattern = r'^([a-zA-Z0-9._-]+)(\[[^\]]*\])?([;#].*)?$'
                match = re.match(extras_pattern, original_line.strip())
                
                if match:
                    pkg = match.group(1)
                    extras = match.group(2) or ''
                    markers = match.group(3) or ''
                    new_line = f"{pkg}{extras}=={latest_version}{markers}"
                else:
                    new_line = f"{package_name}=={latest_version}"
                
                updated_lines.append(new_line)
                changes_made = True
                print(f"✓ Pinned to version {latest_version}")
            else:
                # Could not find version, keep original line
                updated_lines.append(original_line)
                print("✗ Could not determine version")
        else:
            # Keep line as-is (comment, empty line, or already versioned)
            updated_lines.append(original_line)
            if package_name:
                print(f"Skipping {package_name} (already has version constraint)")
    
    # Write output file
    try:
        with open(output_file, 'w') as f:
            for line in updated_lines:
                f.write(line + '\n')
        
        print("-" * 50)
        if changes_made:
            print(f"✓ Updated requirements written to: {output_file}")
        else:
            print("No changes were necessary - all packages already have version constraints")
            
    except Exception as e:
        print(f"Error writing to file {output_file}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Pin dynamic packages in requirements.txt to their latest version as of a specific date'
    )
    parser.add_argument(
        'requirements_file',
        help='Path to requirements.txt file'
    )
    parser.add_argument(
        'date',
        help='Target date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file (default: overwrite input file)',
        default=None
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    
    args = parser.parse_args()
    
    # Parse date
    try:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
        target_date = target_date.replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD format.", file=sys.stderr)
        sys.exit(1)
    
    # Check if date is in the future
    if target_date > datetime.now(timezone.utc):
        print("Warning: Target date is in the future. Results may not be complete.", file=sys.stderr)
    
    # Determine output file
    output_file = args.output if args.output else args.requirements_file
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print("=" * 50)
    
    # Process the requirements file
    if args.dry_run:
        # For dry run, use a temporary approach
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp_name = tmp.name
        process_requirements_file(args.requirements_file, tmp_name, target_date)
        
        # Show diff
        print("\nChanges that would be made:")
        print("=" * 50)
        with open(args.requirements_file, 'r') as orig:
            orig_lines = orig.readlines()
        with open(tmp_name, 'r') as new:
            new_lines = new.readlines()
        
        import difflib
        diff = difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=args.requirements_file,
            tofile=output_file,
            lineterm=''
        )
        for line in diff:
            print(line.rstrip())
        
        import os
        os.unlink(tmp_name)
    else:
        process_requirements_file(args.requirements_file, output_file, target_date)


if __name__ == '__main__':
    main()