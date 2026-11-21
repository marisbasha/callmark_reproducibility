# Energy-threshold detector

A detector without training: a vocalization is a stretch of a chunk whose band-limited
spectrogram energy stands out from the rest of the chunk. The settings are the same for all
corpora except the frequency band.

## Environment

Python 3.11, numpy 2.4.6, soundfile 0.14.0, pyarrow 25.0.0.

## Data

    dataset/segmentation/test-*.parquet     test chunks

## Command

Run from the repository root; `--data` can be replaced by the environment variable
`CALLMARK_DATA`.

    python -m systems.energy.detect --data DIR

Writes `predictions/energy/detection.json.gz`. Run time: about 15 min on one CPU core for the 11,085
test chunks.

## Method

| step | setting |
|---|---|
| audio | channels averaged, sampling rate of the recording |
| spectrogram | Hann window of 1,024 samples, hop of 256 samples; power in dB |
| band | per corpus (table below), upper limit at most half the sampling rate |
| whitening | the median over time of the chunk is subtracted from every frequency |
| frame energy | mean over the band |
| thresholds | median of the frame energy of the chunk plus 3 (start) and plus 1 (end) robust standard deviations (1.4826 x median absolute deviation) |
| vocalization | from the first frame at or above the start threshold to the first frame below the end threshold |
| merging | vocalizations less than 30 ms apart are merged |
| minimum duration | 20 ms |

A chunk shorter than two frames, or with a median absolute deviation of zero, gets no
vocalizations.

| corpus | band (Hz) |
|---|---|
| `budgie` | 0 to 10,000 |
| `canary`, `infant_marmosets`, `nbm`, `ozf` | 0 to 22,050 |
| `enabirds`, `sne_full` | 0 to 16,000 |
| `right_whale` | 0 to 600 |
| `rodent_usv` | 15,000 to 125,000 |
| `singing_mice` | 10,000 to 125,000 |
| `spiders` | 0 to 8,000 |
| `tworadiobirds` | 0 to 12,000 |
| `wytham` | 0 to 11,025 |
