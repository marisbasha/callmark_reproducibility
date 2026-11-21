# CallMark: code to reproduce the experiments

Code for the experiments of the CallMark paper. Each folder is self-contained and has its own
README with the data layout, the commands and the expected output.

| folder | experiment | paper |
|---|---|---|
| `benchmark/` | cross-taxon benchmark of 11 corpora: detection, identification, Detection+Identification and overlap detection, with the statistical tests | benchmark figures and tables |
| `correction/` | boundary correction on synthetic stimuli annotated by eight annotators | boundary-correction figure and table |
| `multichannel/` | vocalization detection with body-mounted sensors (MC-WhisperSeg), and the microphone-only detector trained on group audio labelled by MC-WhisperSeg | Fig. 4, Extended Data figure, Supplementary Table |

## Data

Every script reads one data root, given by the environment variable `CALLMARK_DATA` or by `--data`.

- `correction/` includes its data.
- `multichannel/`: the public group recordings and their annotations are on Zenodo
  (DOI 10.5281/zenodo.13144875). The recordings of pairs used for training and the unannotated
  group recordings are available from the corresponding author.
- `benchmark/`: the dataset and the prediction files are available from the corresponding author
  during peer review and will be archived with the published article.

## Related code

- WhisperSeg: https://github.com/nianlonggu/WhisperSeg
- MCWhisperSeg (the model code in `multichannel/mcwhisperseg/` is modified from it, Apache-2.0):
  https://github.com/nianlonggu/MCWhisperSeg
