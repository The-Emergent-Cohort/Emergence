#!/usr/bin/env python3
"""Download linguistic data sources for concept tokenizer database"""

import argparse
import os
import sys
import zipfile
import shutil
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run: pip install requests tqdm")
    sys.exit(1)


# Data source definitions
SOURCES = {
    'glottolog': {
        'name': 'Glottolog',
        'url': 'https://zenodo.org/api/records/8131084/files/cldf-datasets-glottolog-cldf-v4.8.zip/content',
        'type': 'zenodo_zip',
        'description': 'Language family hierarchy and codes',
        'priority': 1,
    },
    'wals': {
        'name': 'WALS (World Atlas of Language Structures)',
        'url': 'https://zenodo.org/api/records/7385533/files/cldf-datasets-wals-v2020.3.zip/content',
        'type': 'zenodo_zip',
        'description': 'Typological features (word order, morphology type)',
        'priority': 2,
    },
    'grambank': {
        'name': 'Grambank',
        'url': 'https://zenodo.org/api/records/7740139/files/grambank-grambank-v1.0.3.zip/content',
        'type': 'zenodo_zip',
        'description': 'Grammar features (195 features, 2467 languages)',
        'priority': 3,
    },
    'morpholex': {
        'name': 'MorphoLex-en',
        'url': 'https://github.com/hugomailhot/MorphoLex-en/raw/master/MorphoLex_en.xlsx',
        'type': 'direct',
        'filename': 'MorphoLex_en.xlsx',
        'description': 'English morpheme segmentation (70k words)',
        'priority': 5,
    },
}

# UniMorph languages to download (start with high-priority ones)
UNIMORPH_LANGUAGES = [
    'eng',  # English
    'deu',  # German
    'fra',  # French
    'spa',  # Spanish
    'ita',  # Italian
    'por',  # Portuguese
    'rus',  # Russian
    'ara',  # Arabic
    'heb',  # Hebrew
    'jpn',  # Japanese
    'zho',  # Chinese
    'kor',  # Korean
    'hin',  # Hindi
    'tur',  # Turkish
    'pol',  # Polish
    'nld',  # Dutch
    'swe',  # Swedish
    'fin',  # Finnish
    'hun',  # Hungarian
    'ces',  # Czech
]


def download_file(url: str, dest_path: Path, desc: str = None) -> bool:
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc or dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extract zip file and flatten single top-level directory"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)

        # Check if there's a single top-level directory and flatten it
        contents = list(dest_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            top_dir = contents[0]
            for item in top_dir.iterdir():
                shutil.move(str(item), str(dest_dir))
            top_dir.rmdir()

        # Remove the zip file
        zip_path.unlink()

        return True
    except Exception as e:
        print(f"  Error extracting {zip_path}: {e}")
        return False


def download_zenodo_zip(source: dict, dest_dir: Path) -> bool:
    """Download and extract a Zenodo zip file"""
    zip_path = dest_dir / 'download.zip'

    if not download_file(source['url'], zip_path, source['name']):
        return False

    print(f"  Extracting {source['name']}...")
    return extract_zip(zip_path, dest_dir)


def download_direct(source: dict, dest_dir: Path) -> bool:
    """Download a file directly"""
    dest_path = dest_dir / source['filename']
    return download_file(source['url'], dest_path, source['name'])


def download_unimorph(dest_dir: Path, languages: list) -> bool:
    """Download UniMorph data for specified languages"""
    print(f"\n  Downloading UniMorph for {len(languages)} languages...")

    base_url = "https://raw.githubusercontent.com/unimorph/{lang}/master/{lang}"

    success_count = 0
    for lang in tqdm(languages, desc="UniMorph languages"):
        lang_dir = dest_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        # Try common file patterns
        for ext in ['', '.txt']:
            url = base_url.format(lang=lang) + ext
            dest_path = lang_dir / f"{lang}{ext or '.txt'}"

            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    dest_path.write_text(response.text, encoding='utf-8')
                    success_count += 1
                    break
            except:
                continue

    print(f"  Downloaded {success_count}/{len(languages)} UniMorph languages")
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(description='Download linguistic data sources')
    parser.add_argument('--base-dir', type=Path, required=True,
                        help='Base directory for reference data')
    parser.add_argument('--sources', nargs='+', choices=list(SOURCES.keys()) + ['unimorph', 'all'],
                        default=['all'], help='Sources to download')
    parser.add_argument('--unimorph-langs', nargs='+', default=UNIMORPH_LANGUAGES,
                        help='UniMorph languages to download')

    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    sources_to_download = args.sources
    if 'all' in sources_to_download:
        sources_to_download = list(SOURCES.keys()) + ['unimorph']

    # Sort by priority
    sources_to_download = sorted(
        sources_to_download,
        key=lambda s: SOURCES.get(s, {}).get('priority', 99)
    )

    print(f"Downloading to: {base_dir}")
    print(f"Sources: {', '.join(sources_to_download)}")
    print("")

    results = {}

    for source_key in sources_to_download:
        if source_key == 'unimorph':
            print(f"\n[UniMorph] Inflection paradigms...")
            dest_dir = base_dir / 'unimorph'
            results['unimorph'] = download_unimorph(dest_dir, args.unimorph_langs)
        elif source_key in SOURCES:
            source = SOURCES[source_key]
            print(f"\n[{source['name']}] {source['description']}...")
            dest_dir = base_dir / source_key
            dest_dir.mkdir(parents=True, exist_ok=True)

            if source['type'] == 'zenodo_zip':
                results[source_key] = download_zenodo_zip(source, dest_dir)
            elif source['type'] == 'direct':
                results[source_key] = download_direct(source, dest_dir)

    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)

    for source, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {source}")

    # Check directory sizes
    print("\nDirectory sizes:")
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            size = sum(f.stat().st_size for f in subdir.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"  {subdir.name}: {size_mb:.1f} MB")

    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
