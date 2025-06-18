import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from tqdm import tqdm
from xml.etree.ElementTree import ParseError

from elan_data import ELAN_Data

# Constants
FPS: int = 25  # Default frames per second
SAMPLE_RATE: int = 16000
NUM_FOLDS: int = 5
SPEAKER_MAP: Dict[int, str] = {0: 'B', 1: 'C'}

from scipy.signal import savgol_filter

def smoothed_predictions(
    gesture_prob: np.ndarray,
    smoothing_window: int = 30,
    polyorder: int = 3
) -> np.ndarray:
    """
    Smooth a 1D prediction series using a Savitzky-Golay filter.

    Args:
        gesture_prob: Raw gesture probability array.
        smoothing_window: Window length (must be odd and <= len(gesture_prob)).
        polyorder: Polynomial order for smoothing.

    Returns:
        Smoothed prediction array.
    """
    # Ensure odd window length and <= data length
    window = min(smoothing_window, len(gesture_prob))
    if window % 2 == 0:
        window -= 1
    if window < polyorder + 2:
        window = polyorder + 2 + (1 - ((polyorder + 2) % 2))  # make it odd
    window = min(window, len(gesture_prob))
    return savgol_filter(gesture_prob, window_length=window, polyorder=polyorder)
 
def generate_pairs_mappings(max_pair: int = 99) -> Dict[int, str]:
    """
    Generate a mapping from integer pair IDs to zero-padded string representations.

    Example: 1 -> '001', 42 -> '042'.
    """
    return {i: f"{i:03d}" for i in range(1, max_pair + 1)}

PAIRS_MAPPING: Dict[int, str] = generate_pairs_mappings()


def upload_models_result(
    results: List[Dict[str, Any]],
    pairs_mapping: Dict[int, str] = PAIRS_MAPPING,
    num_folds: int = NUM_FOLDS
) -> pd.DataFrame:
    """
    Combine model outputs across folds into a single DataFrame.

    Args:
        results: List of dicts, each containing 'labels', 'preds', 'speaker_ID', 'pair_ID',
                 'start_frames', 'end_frames' arrays for each fold.
        pairs_mapping: Mapping from numeric pair ID to string pair code.
        num_folds: Number of cross-validation folds.

    Returns:
        DataFrame with columns: label, pred_n, pred_g, pair_speaker, start_frame,
        end_frame, fold index.
    """
    accum_preds: np.ndarray = None  # type: ignore
    records: List[Dict[str, Any]] = []

    for fold in range(num_folds):
        data = results[fold]
        # Ensure preds and labels are numpy arrays
        labels: np.ndarray = np.array(data['labels'])
        preds: np.ndarray = np.array(data['preds'])

        # Flatten
        labels_flat = labels.ravel()
        n_samples, seq_len = labels.shape
        preds_flat = preds.reshape(n_samples * seq_len, -1)

        # Expand metadata
        speaker_ids = np.repeat(np.array(data['speaker_ID']), seq_len).ravel()
        pair_ids = np.repeat(np.array(data['pair_ID']), seq_len).ravel()
        start_frames = np.repeat(np.array(data['start_frames']), seq_len).ravel()
        end_frames = np.repeat(np.array(data['end_frames']), seq_len).ravel()

        pair_speakers = [f"{pairs_mapping.get(int(pair), str(pair))}_{SPEAKER_MAP.get(int(s), '?')}"
                         for pair, s in zip(pair_ids, speaker_ids)]

        # Accumulate predictions
        if accum_preds is None:
            accum_preds = preds_flat.copy()
        else:
            accum_preds += preds_flat
        if fold == num_folds - 1:
           for idx in range(len(labels_flat)):
               records.append({
                  'label': int(labels_flat[idx]),
                  'pair_speaker': pair_speakers[idx],
                  'start_frame': int(start_frames[idx]),
                  'end_frame': int(end_frames[idx]),
               })

    # Average predictions over folds
    avg_preds = accum_preds / num_folds

    df = pd.DataFrame(records)
    df[['pred_n', 'pred_g']] = pd.DataFrame(avg_preds.tolist(), index=df.index)
    return df


def divide_segment(
    scores: np.ndarray,
    start: int,
    end: int
) -> List[Tuple[int, int]]:
    """
    Split a segment into sub-segments based on local minima between peaks.
    Returns list of (sub_start, sub_end) indices.

    Args:
        scores: 1D array of prediction scores.
        start: segment start frame (inclusive).
        end: segment end frame (exclusive).
    """
    segment = scores[start:end]
    if len(segment) < 2:
        return [(start, end)]

    minima = find_peaks(-segment, prominence=0.01)[0]
    maxima = find_peaks(segment, prominence=0.01)[0]

    if len(maxima) <= 1 or len(minima) == 0:
        return [(start, end)]

    boundaries: List[Tuple[int, int]] = []
    last = start
    for i, peak in enumerate(maxima[:-1]):
        boundary = minima[i] + start
        boundaries.append((last, boundary))
        last = boundary + 1
    boundaries.append((last, end))
    return boundaries


def save_elan_files(
    elan_template: str,
    media_path: str,
    binary_preds: np.ndarray,
    scores: np.ndarray,
    fps: int = FPS,
    tier_name: str = 'skeleton',
    min_duration_frames: int = 3,
    smoothing_window: int = 30,
    threshold: float = 0.55
) -> None:
    """
    Write segmentation results to an ELAN .eaf file.

    Args:
        elan_template: Path to base .eaf template.
        media_path: Path to media file for linking.
        binary_preds: Binary mask of gesture presence per frame.
        scores: Raw prediction scores per frame.
        fps: Frames per second for time conversion.
        tier_name: Name of ELAN tier to annotate.
        min_duration_frames: Minimum length of a segment to record.
    """
    # check if elan_template exists
    tier_name = f"Threshold-{threshold:.2f}"
    tier_name = f"Threshold-{threshold:.2f}"
    if os.path.exists(elan_template):
        try:
            new_eaf = ELAN_Data.from_file(file=elan_template)
        except ParseError:
            # template is broken XML — make a brand-new one
            new_eaf = ELAN_Data.create_eaf(
                elan_template, audio=media_path,
                tiers=[tier_name], remove_default=True
            )
    else:
        new_eaf = ELAN_Data.create_eaf(
            elan_template, audio=media_path,
            tiers=[tier_name], remove_default=True
        )
    new_eaf.add_tier(tier_name, init_df=False)
    

    i = 0
    length = len(binary_preds)
    scores = smoothed_predictions(scores, smoothing_window=smoothing_window)
    while i < length:
        if not binary_preds[i]:
            i += 1
            continue

        start = i
        while i < length and binary_preds[i]:
            i += 1
        end = i

        if end - start < min_duration_frames:
            continue
         # Divide segment into sub-segments if necessary
        sub_segments = divide_segment(scores, start, end)
        for sub_start, sub_end in sub_segments:
            if sub_end - sub_start < min_duration_frames:
                continue
            start_ms = (sub_start / fps) * 1000
            end_ms = (sub_end / fps) * 1000
            mean_score = float(scores[start:end].mean())
            

            new_eaf.add_segment(
                  tier_name,
                  start=start_ms,
                  stop=end_ms,
                  annotation=f"prob-{mean_score:.2f}"
            )
    new_eaf.save_ELAN(raise_error_if_unmodified=False)


def get_predictions(
    df: pd.DataFrame
) -> np.ndarray:
    """
    Compute softmax-normalized prediction scores.
    Expects columns 'pred_n' and 'pred_g'.
    """
    logits = df[['pred_n', 'pred_g']].to_numpy(dtype=float)
    return softmax(logits, axis=1)


def annotate_predictions(
    df: pd.DataFrame,
    threshold: float = 0.55,
    smoothing_fraction: float = 0.2,
    fps: int = FPS
) -> pd.DataFrame:
    """
    Add binary and continuous gesture predictions to DataFrame.

    Args:
        df: DataFrame with averaged logits.
        threshold: Probability threshold for binary segmentation.
        smoothing_fraction: Fraction of fps to use for median smoothing.
        fps: Frames per second for conversion.
    """
    probs = get_predictions(df)[:, 1]
    binary = probs >= threshold
    size = max(1, int(smoothing_fraction * fps))
    binary_smoothed = median_filter(binary.astype(int), size=size)

    df['gesture_prob'] = probs
    df['gesture_bin'] = binary_smoothed.astype(bool)
    return df


def process_and_save_all(
    results: List[Dict[str, Any]],
    model: str = 'skeleton',
    thresholds: float = 0.55,
    elan_template: str = '',
    media_path: str = '',
    fps: int = FPS,
    smoothing_window: int = 40
) -> None:
    """
    High-level pipeline: upload results, annotate predictions, and save ELAN files per speaker.

    Args:
        results: model outputs per fold.
        model: tier name.
        threshold: gesture detection threshold.
        elan_template: path to ELAN .eaf template.
        media_path: path to media file for linking.
        fps: frames per second (e.g., video fps).
    """
    for threshold in thresholds:
        if not (0 <= threshold <= 1):
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        print(f"Processing threshold: {threshold:.2f}")
        df = upload_models_result(results)
        df = annotate_predictions(df, threshold=threshold, fps=fps)
        df['speaker'] = df['pair_speaker'].str.split('_').str[1]

        for pair_speaker, group in tqdm(df.groupby('pair_speaker')):
            preds = group['gesture_prob'].to_numpy()
            binary = group['gesture_bin'].to_numpy()
            save_elan_files(
                elan_template,
                media_path,
                binary_preds=binary,
                scores=preds,
                fps=fps,
                tier_name=model,
                min_duration_frames=int(fps * 0.11),  # 100 ms minimum segment length
                smoothing_window=smoothing_window,
                threshold=threshold
                
            )
    print(f"ELAN file saved with all thresholds used: {elan_template}")
    return df


if __name__ == '__main__':
   #  Example usage
   results_path = 'Day1/CABB_Segmentation/fold_5/test_results.pkl'
   import pickle
   with open(results_path, 'rb') as f:
      results = pickle.load(f)
   df = process_and_save_all(
      results,
      model='skeleton',
      thresholds=[0.55, 0.60, 0.65],
      elan_template='template.eaf',
      media_path='video.mp4',
      fps=30  # override default if necessary
   )
   pass
