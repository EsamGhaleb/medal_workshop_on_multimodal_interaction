## Automatic Speech Transcription with WhisperX

This tutorial guides you through using **Whisper** (OpenAI's ASR) enhanced by **WhisperX** for high-precision, timestamped speech transcription—ideal for language science research.

---

### 📦 Models & Performance

| Model Size | Parameters | English-only | Multilingual | VRAM ⬇️ | Speed (vs. large) |
| :--------: | :--------: | :----------: | :----------: | :-----: | :---------------: |
|  **tiny**  |    39 M    |   `tiny.en`  |    `tiny`    |  \~1 GB |       \~32×       |
|  **base**  |    74 M    |   `base.en`  |    `base`    |  \~1 GB |       \~16×       |
|  **small** |    244 M   |  `small.en`  |    `small`   |  \~2 GB |        \~6×       |
| **medium** |    769 M   |  `medium.en` |   `medium`   |  \~5 GB |        \~2×       |
|  **large** |   1 550 M  |      N/A     |  `large-v3`  | \~10 GB |         1×        |

> **Tip:** Smaller models run faster but yield lower accuracy. Choose based on your trade-off needs.

---

### ⚙️ Why Use WhisperX?

Whisper’s out‑of‑the‑box transcripts are robust, but:

* **No word‑level timestamps**
* **Imprecise silence handling**

**WhisperX** fixes this by:

1. **Forced alignment** for accurate word (and character) timestamps
2. **Optimized backend** (via faster-whisper)
3. **Customizable ASR options** tailored to linguistic experiments

---

### 📝 Notebook Overview

1. **Setup**

   * Install packages, set paths
2. **Model Loading**

   * Choose model size, device, ASR parameters
3. **Preprocessing**

   * Convert videos → WAV
4. **Transcription + Alignment**

   * Run Whisper, then WhisperX aligner
   * Save JSON with segments + timestamps
5. **Export**

   * TSV (ELAN), TextGrid (Praat), plain-text (WebMAUS), SRT (Captions)

---

### 🔧 Setup & Imports

```python
import whisperx
import torch, os, json
from tqdm import tqdm
from utils.video_converter import extract_wav
from utils.tsv_export import export_transcript_as_tsv
from utils.textgrid_export import export_transcript_as_textgrid
from utils.tsv_export import export_transcript_as_textonly
from utils.srt_export import export_transcript_as_srt
```

*Define your folders:*

```python
INPUT_DIR = './input/'
OUTPUT_DIR = './output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

---

### 🧩 ASR Parameters for Linguistic Analysis

Customize `asr_options` for:

* **Beam size** (`beam_size`): quality vs. speed
* **No speech threshold** (`no_speech_threshold`): controls silence filtering
* **Word timestamps** (`word_timestamps`): `True` for word-level alignment
* **Temperature schedule**: boosts robustness in noisy data

```python
asr_options = {
    'beam_size': 5,
    'no_speech_threshold': 0.6,
    'word_timestamps': True,
    'temperatures': [0.0, 0.2, 0.5, 1.0],
}
model = whisperx.load_model(
    'base', device='cuda', compute_type='float16', asr_options=asr_options
)
```

---

### 🚀 Running the Pipeline

```python
# 1. Convert videos to WAV
for fname in os.listdir(INPUT_DIR):
    extract_wav(os.path.join(INPUT_DIR, fname))

# 2. Transcribe & align
for wav in tqdm(os.listdir(INPUT_DIR)):
    if not wav.endswith('.wav'): continue
    audio = whisperx.load_audio(os.path.join(INPUT_DIR, wav))
    result = model.transcribe(audio)
    # Forced alignment
    align_model, meta = whisperx.load_align_model(
        language_code=result['language'], device='cuda'
    )
    aligned = whisperx.align(
        result['segments'], align_model, meta, audio,
        device='cuda', return_char_alignments=True
    )
    json.dump(aligned, open(f'{OUTPUT_DIR}/{wav}.json','w'), indent=2)
```

---

### 💾 Export Formats

```python
for j in os.listdir(OUTPUT_DIR):
    if j.endswith('.json'):
        data = json.load(open(f'{OUTPUT_DIR}/{j}'))
        basename = j.replace('.json','')
        export_transcript_as_tsv(data, f'{basename}.tsv', OUTPUT_DIR)
        export_transcript_as_textgrid(data, f'{basename}.TextGrid', OUTPUT_DIR)
        export_transcript_as_textonly(data, f'{basename}_plain.txt', OUTPUT_DIR)
        export_transcript_as_srt(f'{basename}.tsv', OUTPUT_DIR)
```

---

## 📝 Exercises

### 1. Import & Compare

* **Import** your `.tsv` into ELAN and TextGrid into Praat.
* **Compare** timestamps: which aligner (ELAN vs. Praat) handles pauses better?

### 2. Model-Size & Parameter Sweep

Modify:

* **`model_size`**: compare `base` vs. `large-v3`
* **`beam_size`**: test values `{1,5,10}`
* **`no_speech_threshold`**: `{0.3,0.6,0.9}`

For each setting:

1. Measure **Word Error Rate** (WER) vs. gold transcript
2. Compute **alignment offset** (mean difference between forced and reference timestamps)
3. Plot **processing time** per 30 s file

### 3. Diarization & Phoneme Analysis

* Enable speaker diarization with `pyannote` backend.
* Extract **phoneme durations** via `return_char_alignments=True`.
* Analyse mean duration of vowels vs. consonants.

---

> **Acknowledgements:** Based on materials from [WhisperX](https://github.com/m-bain/whisperX) and Sho Akamine’s tutorial.
