import tgt
import os
import copy

# import custom-made functions
from utils.time_format_converter import convert_string_to_float
from utils.tsv_export import get_last_phoneme_timestamp


def safe_time(value):
    """
    Convert a raw timestamp (string or number) to float, or return None if not convertible.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def get_tiers(result, sentence_tier, word_tier, puncts, word_spacing):
    # choose spacing
    word_space = " " if word_spacing else ""
    segments = copy.deepcopy(result).get("segments", [])

    # drop unwanted keys
    for seg in segments:
        for k in ("seek", "tokens"):
            seg.pop(k, None)

    for segment in segments:
        utterance = []
        utter_start = None
        last_phoneme_end = None
        words = segment.get("words", [])
        n_words = len(words)

        for idx, word_info in enumerate(words):
            text = word_info.get("word", "")
            raw_start = word_info.get("start")
            # get the end via phoneme timestamp
            raw_end_str = get_last_phoneme_timestamp(segment, word_info, puncts)
            raw_end = convert_string_to_float(raw_end_str) if raw_end_str is not None else None

            start = safe_time(raw_start)
            end = safe_time(raw_end)

            # skip if we have no start or end
            if start is None or end is None:
                print(f"Skipping word with missing timestamps: {word_info}")
                continue

            # add word interval
            word_tier.add_interval(tgt.Interval(start_time=start, end_time=end, text=text))

            # build sentence-level utterance
            if not utterance:
                # first word of a new utterance
                utter_start = start
            utterance.append(text)
            last_phoneme_end = end

            # decide whether to end the utterance here
            is_last_word = (idx == n_words - 1)
            ends_with_punct = text and text[-1] in puncts
            if ends_with_punct or is_last_word:
                sent_text = word_space.join(utterance)
                # add sentence interval
                sentence_tier.add_interval(
                    tgt.Interval(start_time=utter_start, end_time=last_phoneme_end, text=sent_text)
                )
                # reset for next
                utterance = []
                utter_start = None

    return sentence_tier, word_tier


def export_transcript_as_textgrid(result, filename, output_folder, puncts, word_spacing=True):
    # prepare tiers
    final_end = safe_time(result.get("segments", [])[-1].get("end", None))
    if final_end is None:
        print("Cannot determine TextGrid end time; aborting.")
        return

    tg = tgt.TextGrid()
    sentence_tier = tgt.IntervalTier(start_time=0.0, end_time=final_end, name="sentence")
    word_tier     = tgt.IntervalTier(start_time=0.0, end_time=final_end, name="word")

    # fill tiers
    sentence_tier, word_tier = get_tiers(result, sentence_tier, word_tier, puncts, word_spacing)

    # assemble and write
    tg.add_tier(sentence_tier)
    tg.add_tier(word_tier)

    output_name = os.path.splitext(os.path.basename(filename))[0] + ".TextGrid"
    textgrid_dir = os.path.join(os.path.dirname(output_folder), "textgrid")
    os.makedirs(textgrid_dir, exist_ok=True)
    output_path = os.path.join(textgrid_dir, output_name)

    tgt.write_to_file(tg, output_path, format="short")
    print(f"Wrote TextGrid to {output_path}")
