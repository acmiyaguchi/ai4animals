# audio eda

This contains code for doing audio analysis with machine learning techniques for studying animals.

## notes


### uv

Install the packages to a venv located in temporary directory when you can.

```bash
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT=${TMPDIR:-/tmp}/ai4animals/.venv
```

### download

I'll be taking a look at the pacific sounds dataset by MBARI https://registry.opendata.aws/pacific-sound.


```bash
aws s3 ls --no-sign-request s3://pacific-sound-256khz-2025/09
...
2025-10-01 06:11:33  460800310 MARS_20250930_224000.wav
2025-10-01 06:12:06  460800310 MARS_20250930_225000.wav
2025-10-01 06:12:07  460800310 MARS_20250930_230000.wavf
2025-10-01 06:12:09  460800310 MARS_20250930_231000.wav
2025-10-01 06:12:09  460800310 MARS_20250930_232000.wav
2025-10-01 06:12:11  460800310 MARS_20250930_233000.wav
2025-10-01 06:12:44  460800310 MARS_20250930_234000.wav
2025-10-01 06:12:44  460800310 MARS_20250930_235000.wav
```

The soundscapes come in 10 minute intervals.

I'll probably focus on a single soundscape to start with, and extract both embeddings and spectrograms from it.

I choose 10 random files to download from Sept 2025.
I assume that the time of day will affect what kinds of things are in the soundscape.

```bash
~/scratch/ai4animals/audio_eda/pacific_sounds/raw
├── [ 439M]  MARS_20250901_175000.wav
├── [ 439M]  MARS_20250904_064000.wav
├── [ 439M]  MARS_20250905_153000.wav
├── [ 439M]  MARS_20250906_230000.wav
├── [ 439M]  MARS_20250907_135000.wav
├── [ 439M]  MARS_20250915_190000.wav
├── [ 439M]  MARS_20250916_112000.wav
├── [ 439M]  MARS_20250920_080000.wav
├── [ 439M]  MARS_20250927_034000.wav
├── [ 439M]  MARS_20250927_163000.wav
├── [    0]  _SUCCESS
└── [  250]  filenames.txt
```

These files are pretty big actually.

### embed

On a gpu instance, I run the audio through a birdcall embedding model to get embeddings for 5 second segments at 2hz resolution.

This takes about 30 minutes total when split across 5 tasks on a L40s. 

### eda

Now we do some light exploratory analysis.
