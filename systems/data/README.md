# Data preparation shared by the benchmark systems

These scripts build, from the published dataset, the prepared detection audio and the label files
of the data root described in `benchmark/README.md`.

## Environment

Python 3.11 with numpy 2.4.6, pyarrow 25.0.0 and soundfile 0.14.0 (libsndfile 1.2.2). Run from
the repository root. `--data` can be replaced by the environment variable `CALLMARK_DATA`.

## Input

    dataset/classification/{train_500,test}-*.parquet
    dataset/segmentation/{train_500,test}-*.parquet

Rows of the corpus `rodents2019` are ignored. The recordings of right whales and spiders are
prepared like the 11 scored corpora.

## Commands

    python -m systems.data.build_labels --data DIR
    python -m systems.data.prepare_test --data DIR
    python -m systems.data.prepare_train --data DIR
    python -m systems.data.join_windows --data DIR

| command | writes | content | run time (12 processes) |
|---|---|---|---|
| `build_labels` | `labels/test_taxa.json`, `labels/end_to_end_chunks.json` | taxon of every test annotation; 1,315 chunks of the Detection+Identification subset | 1 s |
| `prepare_test` | `prepared/test/<chunk_id>.wav`, `.json` | 11,085 test chunks | 1 min |
| `prepare_train` | `prepared/train_<size>/<chunk_id>__w<number>.wav`, `.json` | training windows of sizes 50, 100 and 500 | 1 min |
| `join_windows` | `prepared/train_<size>_joined/<corpus>__g<group>_cat<number>.wav`, `.json` | the same windows concatenated into files of at least 30 s | 1 min |

`labels/excluded_long_test.json` is shipped with the data: it lists vocalizations that were
removed when the dataset was built (longer than the 95th percentile of their corpus), which can
only be read from the original corpora.

## Prepared files

Audio is 16-bit WAV at the sampling rate of the recording. Every JSON file holds

| key | content |
|---|---|
| `onset`, `offset` | annotations in seconds from the start of the file |
| `cluster` | test: `"Vocal"` for every annotation; training: the taxon, `"unknown"` if unidentified |
| `species` | name of the corpus |
| `sr` | sampling rate (Hz) |
| `min_frequency` | 0 Hz; `singing_mice` 10,000 Hz; `rodent_usv` 15,000 Hz |
| `spec_time_step` | 0.0025 s; 0.0005 s at sampling rates of 200 kHz and above |
| `min_segment_length` | 0.01 s; 0.005 s at 200 kHz and above |
| `eps` | 0.02 s; 0.005 s at 200 kHz and above |
| `time_per_frame_for_scoring` | 0.001 s; 0.0002 s at 200 kHz and above |
| `tolerance` | 0.01 s |

Test annotations have offsets clipped to the end of the audio and times rounded to 1 µs; times
of training windows are rounded to 0.1 ms.

## Taxon of an annotation

The detection chunks of the dataset carry no taxa. An annotation takes the taxon of its own
identification example: the example of the same chunk whose onset and offset agree (summed
difference at most 2 ms), each annotation used once. An annotation without an example takes the
taxon of its corpus if the corpus has a single taxon (`budgie`, `canary`, `infant_marmosets`,
`ozf`, `right_whale`, `singing_mice`, `spiders`, `tworadiobirds`, `wytham`), in `rodent_usv` the
only taxon among the examples of its recording, and stays unidentified in `enabirds`, `nbm` and
`sne_full`.

## Detection+Identification subset

Per corpus, the test chunk identifiers are sorted; a corpus with at most 300 chunks keeps all,
otherwise chunk `int(i * n / 300)` for `i = 0 ... 299` of its `n` chunks is kept (`budgie` and
`wytham`).

## Training windows

A training set of size N holds at most N annotated vocalizations of every taxon of every corpus.

1. Annotations are those of the `train_500` chunks, with offsets clipped to the end of the
   audio; an annotation with less than 1 ms inside the audio is not used.
2. Pairs of taxon and corpus are processed rarest first (fewest annotations in `train_500`). The
   identification examples of a pair are taken by increasing `fewshot_rank` until the pair holds
   N annotated vocalizations in the windows accepted so far.
3. An example seeds a window around its vocalization. The window is widened until no annotation
   crosses an edge or lies within 1 ms of it. It then grows by one annotation at a time, nearest
   centre first, and stops at the first step that would make it longer than 5 s or take a pair
   above N. A seed whose first window takes a pair above N is skipped.
4. The window is extended over the vocalization-free audio on both sides, up to 5 ms from the
   next annotation or to the start or end of the chunk.
5. Overlapping windows of a chunk are merged and cut from the chunk on whole samples.

Unidentified vocalizations inside a window are annotated (`"unknown"`) and not counted.

| size | windows | chunks | hours | annotations | identified | joined files |
|---|---|---|---|---|---|---|
| 50 | 1,333 | 1,064 | 3.45 | 5,625 | 4,611 | 315 |
| 100 | 1,925 | 1,428 | 5.59 | 9,269 | 7,480 | 503 |
| 500 | 3,237 | 2,250 | 15.33 | 24,957 | 18,530 | 1,203 |

## Joined windows

Windows of one corpus with the same sampling rate, `spec_time_step` and `min_frequency` form a
group and are concatenated in name order without gaps. A file is closed once it reaches 30 s; a
shorter remainder is appended to the last file of the group. The annotations and their taxa are
those of the windows.
