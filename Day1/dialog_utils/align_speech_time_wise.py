

from typing import List, Union, Iterator

import numpy as np
from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.alignment import get_trellis, backtrack, merge_repeats, merge_words
from whisperx.utils import format_timestamp

import torch
from tqdm import tqdm

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
}

def get_audio(audio: Union[str, np.ndarray, torch.Tensor, dict], speaker: str):
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        if isinstance(audio, dict):
            return audio[speaker]
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    return audio

def align(
    transcript: Union[Iterator[dict], List[dict]],
    model: torch.nn.Module,
    align_model_metadata: dict,
    orig_audio: Union[str, np.ndarray, torch.Tensor, dict],
    device: str,
    extend_duration: float = 0.0,
    start_from_previous: bool = True,
    drop_non_aligned_words: bool = False,
):
   #  print("Performing alignment...")


    model_dictionary = align_model_metadata['dictionary']
    model_lang = align_model_metadata['language']
    model_type = align_model_metadata['type']

    prev_t2 = 0
    word_segments_list = []
    for idx, segment in enumerate(tqdm(transcript)):
        audio = get_audio(orig_audio, segment['speaker'])
        MAX_DURATION = audio.shape[0] / SAMPLE_RATE

        t1 = max(segment['start'] - extend_duration, 0)
        t2 = min(segment['end'] + extend_duration, MAX_DURATION)
        # if start_from_previous and t1 < prev_t2:
        #     t1 = prev_t2

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)
      #   print(idx)
        waveform_segment = audio[f1:f2]
        # convert waveform to torch tensor
        waveform_segment = torch.from_numpy(waveform_segment).unsqueeze(0)
        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device))
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(device)).logits
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")
            emissions = torch.log_softmax(emissions, dim=-1)
        emission = emissions[0].cpu().detach()
        transcription = segment['text'].strip()
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            t_words = transcription.split(' ')
        else:
            t_words = [c for c in transcription]

        t_words_clean = [''.join([w for w in word if w.lower() in model_dictionary.keys()]) for word in t_words]
        t_words_nonempty = [x for x in t_words_clean if x != ""]
        t_words_nonempty_idx = [x for x in range(len(t_words_clean)) if t_words_clean[x] != ""]
        segment['word-level'] = []

        if len(t_words_nonempty) > 0:
            transcription_cleaned = "|".join(t_words_nonempty).lower()
            tokens = [model_dictionary[c] for c in transcription_cleaned]
            trellis = get_trellis(emission, tokens)
            try:
                path = backtrack(trellis, emission, tokens)
            except:
                segment['word-level'].append({"text": segment['text'], "start": segment['start'], "end":segment['end']})
                word_segments_list.append({"text": segment['text'], "start": segment['start'], "end":segment['end']})
                print('*************')
                continue
            segments = merge_repeats(path, transcription_cleaned)
            word_segments = merge_words(segments)
            ratio = waveform_segment.size(0) / (trellis.size(0) - 1)

            duration = t2 - t1
            local = []
            t_local = [None] * len(t_words)
            for wdx, word in enumerate(word_segments):
                t1_ = ratio * word.start
                t2_ = ratio * word.end
                local.append((t1_, t2_))
                t_local[t_words_nonempty_idx[wdx]] = (t1_ * duration + t1, t2_ * duration + t1)
            t1_actual = t1 + local[0][0] * duration
            t2_actual = t1 + local[-1][1] * duration

            segment['start'] = t1_actual
            segment['end'] = t2_actual
            prev_t2 = segment['end']

            # for the .ass output
            for x in range(len(t_local)):
                curr_word = t_words[x]
                curr_timestamp = t_local[x]
                if curr_timestamp is not None:
                    segment['word-level'].append({"text": curr_word, "start": curr_timestamp[0], "end": curr_timestamp[1]})
                else:
                    segment['word-level'].append({"text": curr_word, "start": None, "end": None})

            # for per-word .srt ouput
            # merge missing words to previous, or merge with next word ahead if idx == 0
            for x in range(len(t_local)):
                curr_word = t_words[x]
                curr_timestamp = t_local[x]
                if curr_timestamp is not None:
                    word_segments_list.append({"text": curr_word, "start": curr_timestamp[0], "end": curr_timestamp[1]})
                elif not drop_non_aligned_words:
                    # then we merge
                    if x == 0:
                        t_words[x+1] = " ".join([curr_word, t_words[x+1]])
                    else:
                        word_segments_list[-1]['text'] += ' ' + curr_word
        else:
            # then we resort back to original whisper timestamps
            # segment['start] and segment['end'] are unchanged
            prev_t2 = 0
            segment['word-level'].append({"text": segment['text'], "start": segment['start'], "end":segment['end']})
            word_segments_list.append({"text": segment['text'], "start": segment['start'], "end":segment['end']})

        print(f"[{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}] {segment['text']}")
        print(f"  {segment['word-level']}")
    return {"segments": transcript, "word_segments": word_segments_list}