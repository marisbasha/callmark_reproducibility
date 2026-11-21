"""Download the XCL configuration of BirdSet and the eBird taxonomy.

    python -m systems.sources.download_birdset_xcl --data DIR

Writes, under birdset_xcl/ of the data root,

    XCL/XCL_metadata.parquet, XCL/XCL_shard_*.tar.gz    BirdSet XCL (98 archives, 520 GB)
    ebird_taxonomy.csv                                  species of the eBird taxonomy of 2025

The recordings themselves are extracted from the archives by
systems.whisperid.pretraining.build_xcl_manifest.
"""
import argparse

import requests
from huggingface_hub import snapshot_download

from benchmark import config

BIRDSET = "DBD-research-group/BirdSet"
BIRDSET_BRANCH = "data"
TAXONOMY_URL = "https://api.ebird.org/v2/ref/taxonomy/ebird"
TAXONOMY_VERSION = "2025"


def download_taxonomy(path):
    """Write the species rows of the eBird taxonomy (columns SPECIES_CODE, SCIENTIFIC_NAME and
    others)."""
    parameters = {"fmt": "csv", "cat": "species", "version": TAXONOMY_VERSION}
    response = requests.get(TAXONOMY_URL, params=parameters, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)


def main():
    """Download the metadata, the audio archives and the taxonomy."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    directory = config.DATA / "birdset_xcl"
    snapshot_download(BIRDSET, repo_type="dataset", revision=BIRDSET_BRANCH,
                      allow_patterns=["XCL/*"], local_dir=str(directory), max_workers=8)
    download_taxonomy(directory / "ebird_taxonomy.csv")


if __name__ == "__main__":
    main()
