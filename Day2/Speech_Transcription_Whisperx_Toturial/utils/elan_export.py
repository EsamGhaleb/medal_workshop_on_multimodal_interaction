import os
import numpy as np
from elan_data import ELAN_Data
from xml.etree.ElementTree import ParseError
from tqdm import tqdm

def export_json_to_elan(
    json_data: dict,
    elan_output: str,
    media_path: str,
    sentence_tier_name: str = "sentence",
    word_tier_name: str = "word",
    export_word_level_annotations: bool = True,
) -> None:
    """
    Write JSON transcription to an ELAN .eaf file with two tiers: sentences and words.

    Args:
        json_data: Dict containing "segments" list with keys "start", "end", "text", "words".
        elan_output: Path where the new .eaf should be saved.
        media_path: Path to the media file to link.
        fps: Frames per second for time-to-ms conversion (if needed).
        sentence_tier_name: Name of the sentence tier.
        word_tier_name: Name of the word tier.
    """
    # load or create base EAF
    if os.path.exists(elan_output):
        try:
            eaf = ELAN_Data.from_file(file=elan_output)
        except ParseError:
            eaf = ELAN_Data.create_eaf(elan_output, audio=media_path,
                                       tiers=[sentence_tier_name, word_tier_name],
                                       remove_default=True)
    else:
        eaf = ELAN_Data.create_eaf(elan_output, audio=media_path,
                                   tiers=[sentence_tier_name, word_tier_name],
                                   remove_default=True)

    # ensure tiers exist
    eaf.add_tier(sentence_tier_name, init_df=False)
    eaf.add_tier(word_tier_name, init_df=False)
    # link the media
    eaf.add_audio(media_path)

    for seg in tqdm(json_data.get("segments", []), desc="Processing segments"):
        # sentence-level annotation
        start_s = float(seg["start"])
        end_s   = float(seg["end"])
        start_ms = start_s * 1000
        end_ms   = end_s * 1000
        sent_text = seg.get("text", "").strip()

        eaf.add_segment(
            tier=sentence_tier_name,
            start=start_ms,
            stop=end_ms,
            annotation=sent_text
        )
        # word-level annotations
        if export_word_level_annotations:
            for w in seg.get("words", []):
                  w_start = w.get("start")
                  w_end   = w.get("end")
                  # skip if missing timing
                  if w_start is None or w_end is None:
                     continue
                  ws_ms = float(w_start) * 1000
                  we_ms = float(w_end)   * 1000
                  w_text = w.get("word", "")                  

                  eaf.add_segment(
                     tier=word_tier_name,
                     start=ws_ms,
                     stop=we_ms,
                     annotation=w_text
                  )

    # save out
    eaf.save_ELAN(raise_error_if_unmodified=False)

if __name__ == "__main__":
    # Example usage
    result_aligned = {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Hello world",
                "words": [
                    {"start": 0.0, "end": 1.0, "word": "Hello", "score": 0.95},
                    {"start": 1.0, "end": 2.0, "word": "world", "score": 0.90}
                ]
            },
            {
                "start": 5.0,
                "end": 10.0,
                "text": "This is a test",
                "words": [
                    {"start": 5.0, "end": 6.0, "word": "This", "score": 0.92},
                    {"start": 6.0, "end": 7.0, "word": "is",   "score": 0.88},
                    {"start": 7.0, "end": 8.0, "word": "a",    "score": 0.85},
                    {"start": 8.0, "end": 10.0, "word": "test",   "score": 0.91}
                ]
            }
        ]
    }

    