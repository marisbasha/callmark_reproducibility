# External inputs: original corpora and pretraining audio

These scripts obtain the inputs that the other folders need and that are not part of the
published dataset:

- the original corpora of the benchmark, from which the published dataset and two label files
  are built;
- BirdSet XCL and the segmented xeno-canto recordings, the pretraining audio of WhisperSeg
  (`systems/whisperseg`) and WhisperID (`systems/whisperid`).

Run every command from the repository root. `--data` can be replaced by the environment variable
`CALLMARK_DATA` (data root of `benchmark/README.md`).

## Environment

Python 3.11 with numpy 2.4.6, pandas 2.3.3, pyarrow 25.0.0, soundfile 0.14.0 (libsndfile 1.2.2),
datasets 5.0.1, huggingface_hub 0.36.2 and requests 2.34.2. The xeno-canto scripts also need
mutagen 1.48.1 and librosa 0.11.0, and `xeno_canto_segments` the environment of WhisperSeg
prediction (`systems/whisperseg/README.md`).

## Files

| file | content |
|---|---|
| `download_corpora.py` | downloads the public original corpora |
| `corpora.py` | one reader per corpus: recordings and annotations |
| `build.py` | duration filter, chunks, split, identification examples |
| `build_dataset.py` | writes the dataset (parquet files) |
| `build_labels.py` | writes the two label files that need the original annotations |
| `download_birdset_xcl.py` | downloads BirdSet XCL and the eBird taxonomy |
| `xeno_canto_metadata.py`, `xeno_canto_recordings.py`, `xeno_canto_segments.py` | list, select and segment the xeno-canto recordings |

## Folders of the data root

    corpora/<name>/                             original corpora (below)
    corpora/_archives/                          downloaded archives
    birdset_xcl/XCL/XCL_metadata.parquet        metadata of BirdSet XCL
    birdset_xcl/XCL/XCL_shard_*.tar.gz          audio archives of BirdSet XCL
    birdset_xcl/ebird_taxonomy.csv              species of the eBird taxonomy
    xeno_canto/metadata.jsonl                   records of the xeno-canto recordings
    xeno_canto/recordings/                      their audio files
    xeno_canto/selected_recordings.jsonl        recordings that are segmented
    xeno_canto/segments.jsonl                   detected vocalizations
    labels/excluded_long_test.json
    labels/train_500_source_annotations.json
    dataset/{classification,segmentation}/      the dataset, when it is built here

## Original corpora

    python -m systems.sources.download_corpora --data DIR
    python -m systems.sources.corpora --data DIR        # recordings, vocalizations, taxa per corpus

| corpus | content | origin | licence | files read from `corpora/<name>/` |
|---|---|---|---|---|
| `budgie` | budgerigars, 20 kHz | in-house recordings, not in a public archive | CC BY 4.0 | a table `<name>.csv` (comma-separated, columns `onset`, `offset`) next to every `<name>.wav`, in any subfolder |
| `singing_mice` | Alston's singing mice, 250 kHz | in-house recordings, not in a public archive | CC BY 4.0 | a table `<name>.csv` (semicolon-separated, columns `onset`, `offset`) next to every `<name>.wav` |
| `canary` | one domestic canary | Zenodo, https://doi.org/10.5281/zenodo.6521932 | CC BY 4.0 | `audio/*.wav`, `audacity-annotations/*.audacity.txt` |
| `enabirds` | Eastern North American forest birds | Zenodo record 4656848, https://doi.org/10.5061/dryad.d2547d81z | CC0 1.0 | `Recording_<n>/<name>.wav`, `<name>.Table.1.selections.txt` |
| `infant_marmosets` | InfantMarmosetsVox | Zenodo record 10130104, https://doi.org/10.34777/twy8-6468 | CC BY 4.0 | `InfantMarmosetsVox/labels.csv`, `InfantMarmosetsVox/data/twin_<n>/*.wav` |
| `nbm` | nocturnal bird migration in Europe | Zenodo, https://doi.org/10.5281/zenodo.17573913 | recordings of the authors CC BY 4.0; xeno-canto recordings under the licence of each recording | `train_nbm_xc/*.wav`, `test/*.wav`, a label file `<name>.txt` next to each |
| `ozf` | group of eight zebra finches | Voxaboxen data, Zenodo, https://doi.org/10.5281/zenodo.15507508 (`OZF.zip`) | CC BY 4.0 | `OZF/formatted/audio/*.flac`, `OZF/formatted/selection_tables/*.txt` |
| `right_whale` | North Atlantic right whales | FASD13, Zenodo, https://doi.org/10.5281/zenodo.15843741 (`RW.zip`) | CC BY 4.0 | `<name>.wav`, `<name>.csv` |
| `spiders` | jumping spiders | FASD13, Zenodo, https://doi.org/10.5281/zenodo.15843741 (`JS.zip`) | CC BY-SA 4.0 | `<name>.wav`, `<name>.csv` |
| `rodent_usv` | ultrasonic vocalizations of mice, rats and gerbils | USVSEG data, Zenodo, https://doi.org/10.5281/zenodo.3428024 | CC BY 4.0 | `<group>/<name>.wav`, `<name>.csv` |
| `sne_full` | soundscapes of the southern Sierra Nevada | Zenodo, https://doi.org/10.5281/zenodo.7525805 | CC BY 4.0 | `annotations.csv`, `species.csv`, `*.flac` |
| `tworadiobirds` | pairs of zebra finches with radio transmitters | Zenodo, https://doi.org/10.5281/zenodo.7253729 | CC BY 4.0 | `*_voc.csv` and the `*.w64` files they name, in any subfolder |
| `wytham` | great tits of Wytham Woods | Open Science Framework, https://osf.io/n8ac9 | none stated | `crowsetta.csv` (`great-tit-hits-crowsetta.csv` of the record), `WAV/*.wav` |

- `download_corpora` downloads and unpacks all corpora with a public origin (about 65 GB of
  archives). It renames the `.WAV` files of `enabirds` to `.wav` and moves the content of the
  top folder of the `nbm`, `spiders` and `right_whale` archives into `corpora/<name>/`.
- The corpus `rodents2019` is read from `corpora/rodent_usv/`: the same recordings with every
  vocalization labelled as house mouse. It is ignored wherever the dataset is read and is part
  of the build because its vocalizations share the split and the caps of the house mouse.
- Labels that are not vocalizations are dropped by the readers: `SIL` (`canary`), call types 11
  (silence) and 12 (noise) (`infant_marmosets`), rows with `Q` other than `POS` (`spiders`,
  `right_whale`), the label `Other` (`nbm`). `nbm` uses the subsets `train_nbm_xc` and `test`.
- Vocalizations without taxon: the 18 four-letter codes of `enabirds` outside the 30 taxa of
  the benchmark, and the code `????` of `sne_full` (named `unknown`).

## Label files

    python -m systems.sources.build_labels --data DIR

Reads `corpora/` and `dataset/segmentation`.

| file | content |
|---|---|
| `labels/excluded_long_test.json` | `{chunk_id: [[onset, offset, taxon], ...]}`: vocalizations removed by the duration filter that overlap a test chunk, clipped to the chunk (3,303 vocalizations in 1,395 chunks) |
| `labels/train_500_source_annotations.json` | `{chunk_id: [[onset, offset], ...]}` for every `train_500` chunk: all annotations of the original recording that overlap the chunk (listed up to 1 ms outside it, not clipped), including the removed long vocalizations |

Times are seconds from the start of the chunk.

## Dataset

    python -m systems.sources.build_dataset --data DIR

Builds `dataset/classification` and `dataset/segmentation` from `corpora/` (all 13 folders are
needed). The published dataset is the output of these rules:

| step | rule |
|---|---|
| duration filter | per corpus, a vocalization longer than the 95th percentile of the identified vocalizations is removed (also a vocalization without taxon) |
| chunks | cuts every 60 s from the start of the recording to its last offset; a cut inside a vocalization moves to the nearer edge of that vocalization; a chunk without an onset is dropped; a chunk ends at the last offset of its vocalizations if that is later than the cut; times rounded to 0.1 ms |
| split | whole recordings. Single-taxon corpora: recordings largest first into the test set while they fit into the remaining 20 % of the vocalizations of the taxon (15 % slack). Other corpora: recordings with most taxa first; a recording goes to the test set when it fills more of the missing test vocalizations of its taxa than it exceeds them |
| identification examples | per taxon and side, the vocalizations in random order (seed 42 and the name of the taxon), at most 500; the position is `fewshot_rank` |
| `scorable` | taxon in at least two recordings with at least 25 training and 10 test examples |
| identification splits | `train_<N>`: examples of rank below N; `test`: all test examples; 5 s of audio centred on the vocalization |
| detection splits | `test`: every chunk of the test recordings; `train_<N>`: chunks with an example of rank below N |
| audio | 16-bit FLAC at the sampling rate of the recording, mean of the channels |

A taxon that occurs in several corpora (for example the zebra finch in `ozf` and
`tworadiobirds`, the great tit in `wytham` and `nbm`) has one test share and one set of 500
examples per side over all of them.

Longest duration kept (s): `budgie` 0.5100, `canary` 1.5160, `enabirds` 2.8204,
`infant_marmosets` 1.0376, `nbm` 0.9514, `ozf` 0.1555, `right_whale` 1.0000, `rodent_usv`
0.1677, `rodents2019` 0.1681, `singing_mice` 0.1152, `sne_full` 2.3000, `spiders` 0.1970,
`tworadiobirds` 0.1120, `wytham` 0.1858.

## BirdSet XCL

    python -m systems.sources.download_birdset_xcl --data DIR

| input | origin |
|---|---|
| `birdset_xcl/XCL/` | configuration XCL of BirdSet, https://huggingface.co/datasets/DBD-research-group/BirdSet, branch `data` (commit `806ed2cda4ddcbe6efa194ccafff930aa0e557ce`): 528,422 xeno-canto recordings at 32 kHz with detected events, 98 archives, 520 GB |
| `birdset_xcl/ebird_taxonomy.csv` | eBird taxonomy of 2025, species only, from `https://api.ebird.org/v2/ref/taxonomy/ebird` (`SPECIES_CODE` to `SCIENTIFIC_NAME`) |

The recordings used for pretraining are chosen and extracted to `birdset_xcl/audio/` by
`python -m systems.whisperid.pretraining.build_xcl_manifest`: the recordings of the 128 bird
taxa of the identification training set that the taxonomy names, without the recordings of the
benchmark (99,170 recordings, 994,479 detected events of 0.02 to 5 s). WhisperSeg pretraining
uses the same extracted recordings.

## Xeno-canto recordings with WhisperSeg segments

    python -m systems.sources.xeno_canto_metadata --data DIR
    python -m systems.sources.xeno_canto_recordings --data DIR
    python -m systems.sources.xeno_canto_segments --data DIR --model CHECKPOINT

| step | reads | writes |
|---|---|---|
| `xeno_canto_metadata` | VoCallBase collection, https://dataplatform.evolvinglanguage.ch/animal_call | `xeno_canto/metadata.jsonl`: per species the records of its recordings |
| `xeno_canto_recordings` | `metadata.jsonl`, `xeno_canto/recordings/` | `xeno_canto/selected_recordings.jsonl`: at most 50 recordings per species |
| `xeno_canto_segments` | `selected_recordings.jsonl`, `xeno_canto/recordings/` | `xeno_canto/segments.jsonl`, read by `systems.whisperid.pretraining.build_xc_manifest` |

- `xeno_canto/recordings/` holds the audio files of the collection under the names in the field
  `file_original_name` of the records. They are copied from the storage of the collection;
  there is no download script for them.
- Selection: per species, the recordings found in `xeno_canto/recordings/` are shuffled (numpy
  seed 100, species in the order of `metadata.jsonl`); a fifth of them (at least one) is set
  aside and the first 50 of the others are taken. The only recording of a species with a single
  recording is taken.
- Segmentation: every recording is read at its own sampling rate and segmented with the default
  settings of `WhisperSegmenterFast.segment`. WhisperSeg is imported from an unmodified clone:

      git clone https://github.com/nianlonggu/WhisperSeg systems/sources/WhisperSeg

- The stored segments were made with a WhisperSeg model fine-tuned for the detection of animal
  vocalizations that is not public. Public checkpoints of WhisperSeg are listed in its
  repository.
