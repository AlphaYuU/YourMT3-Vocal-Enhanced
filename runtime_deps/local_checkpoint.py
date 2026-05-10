"""Local model.pt inference for the r47b single-source route."""

from __future__ import annotations

import os
import json
import shutil
import site
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable
from uuid import uuid4

PROJECT_ROOT = Path(os.environ.get("R47B_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
TRAIN_SRC = Path(os.environ.get("R47B_YOURMT3_SRC", PROJECT_ROOT / "yourmt3_train_src")).resolve()
CACHE_ROOT = Path(os.environ.get("R47B_CACHE_ROOT", PROJECT_ROOT / ".cache")).resolve()

for key, rel in {
    "HF_HOME": "hf",
    "HF_HUB_CACHE": "hf",
    "HF_ASSETS_CACHE": "hf/assets",
    "HF_DATASETS_CACHE": "hf/datasets",
    "TRANSFORMERS_CACHE": "hf/transformers",
    "TORCH_HOME": "torch",
    "XDG_CACHE_HOME": "xdg",
    "WANDB_DIR": "wandb",
    "WANDB_CACHE_DIR": "wandb/cache",
    "WANDB_CONFIG_DIR": "wandb/config",
    "MPLCONFIGDIR": "mpl",
    "NUMBA_CACHE_DIR": "numba",
    "PIP_CACHE_DIR": "pip",
    "TMP": "tmp",
    "TEMP": "tmp",
    "TMPDIR": "tmp",
    "PYTHONPYCACHEPREFIX": "pycache",
}.items():
    path = CACHE_ROOT / rel
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(key, str(path))
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

site.addsitedir(str(PROJECT_ROOT))
site.addsitedir(str(TRAIN_SRC))

import librosa
import mido
import numpy as np
import pretty_midi
import torch
import torchaudio

torch.compile = lambda fn=None, **kwargs: (fn if fn is not None else (lambda f: f))

from config.vocabulary import program_vocab_presets
from model.ymt3 import YourMT3
from utils.audio import slice_padded_array
from utils.event2note import merge_zipped_note_events_and_ties_to_notes
from utils.note2event import mix_notes
from utils.task_manager import TaskManager
from utils.utils import create_inverse_vocab, write_model_output_as_midi


def load_local_model(model_path: Path, config_path: Path, device: str) -> YourMT3:
    print(f"[r47b model] loading weights: {model_path}", flush=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
    task_name = config["task_manager"]["task_name"]
    max_shift = int(config["task_manager"]["max_shift_steps"])
    task_manager = TaskManager(task_name=task_name, max_shift_steps=max_shift)
    kwargs = dict(config.get("model_init_kwargs") or {})
    model = YourMT3(
        audio_cfg=config.get("audio_cfg"),
        model_cfg=config.get("model_cfg"),
        shared_cfg=config.get("shared_cfg"),
        task_manager=task_manager,
        eval_vocab=config.get("eval_vocab"),
        eval_drum_vocab=config.get("eval_drum_vocab"),
        eval_subtask_key=config.get("eval_subtask_key", "default"),
        onset_tolerance=config.get("onset_tolerance", 0.05),
        write_output_vocab=config.get("write_output_vocab"),
        **kwargs,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    load_rules = config.get("load_state_dict") or {}
    allowed_missing = tuple(load_rules.get("allowed_missing_prefixes") or ["aux_", "f0_adapter"])
    allowed_unexpected = tuple(load_rules.get("allowed_unexpected_prefixes") or ["pitchshift."])
    bad_missing = [key for key in missing if not key.startswith(allowed_missing)]
    bad_unexpected = [key for key in unexpected if not key.startswith(allowed_unexpected)]
    print(f"[r47b model] state load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"state mismatch: bad_missing={bad_missing[:8]} bad_unexpected={bad_unexpected[:8]}")

    if not hasattr(model, "midi_output_inverse_vocab"):
        model.midi_output_vocab = config.get("write_output_vocab") or program_vocab_presets["gm_ext_plus"]
        model.midi_output_inverse_vocab = create_inverse_vocab(model.midi_output_vocab)
    model.to(device)
    model.eval()
    return model


def split_audio(audio: np.ndarray, sr: int, chunk_seconds: float, overlap_seconds: float) -> list[tuple[float, np.ndarray]]:
    if chunk_seconds <= 0:
        return [(0.0, audio)]
    chunk_samples = max(1, int(round(chunk_seconds * sr)))
    overlap_samples = max(0, int(round(overlap_seconds * sr)))
    hop_samples = max(1, chunk_samples - overlap_samples)
    segments = []
    start = 0
    while start < len(audio):
        end = min(len(audio), start + chunk_samples)
        segments.append((start / sr, audio[start:end]))
        if end >= len(audio):
            break
        start += hop_samples
    return segments


def preprocess_audio(model: YourMT3, audio: np.ndarray, sr: int) -> torch.Tensor:
    audio_tensor = torch.from_numpy(audio.astype("float32")).unsqueeze(0)
    target_sr = int(model.audio_cfg["sample_rate"])
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    input_frames = int(model.audio_cfg["input_frames"])
    audio_segments = slice_padded_array(audio_tensor.numpy(), input_frames, input_frames)
    return torch.from_numpy(audio_segments.astype("float32")).unsqueeze(1)


def decode_outputs_to_mido(model: YourMT3, outputs, work_root: Path) -> mido.MidiFile:
    pred_token_arr = outputs
    total_segments = sum(arr.shape[0] for arr in pred_token_arr)
    input_frames = model.audio_cfg["input_frames"]
    sample_rate = model.audio_cfg["sample_rate"]
    start_secs_file = [input_frames * i / sample_rate for i in range(total_segments)]
    num_channels = model.task_manager.num_decoding_channels
    pred_notes_in_file = []
    error_count = Counter()
    for ch in range(num_channels):
        pred_token_arr_ch = [arr[:, ch, :] for arr in pred_token_arr]
        zipped_note_events_and_tie, _list_events, ne_err_cnt = model.task_manager.detokenize_list_batches(
            pred_token_arr_ch,
            start_secs_file,
            return_events=True,
        )
        pred_notes_ch, n_err_cnt_ch = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
        pred_notes_in_file.append(pred_notes_ch)
        error_count += ne_err_cnt
        error_count += n_err_cnt_ch
    pred_notes = mix_notes(pred_notes_in_file)

    work_dir = work_root / f"decode_{uuid4().hex}"
    try:
        track_name = "yourmt3_output"
        write_model_output_as_midi(pred_notes, str(work_dir), track_name, model.midi_output_inverse_vocab)
        midi_path = work_dir / "model_output" / f"{track_name}.mid"
        if not midi_path.exists():
            raise FileNotFoundError(midi_path)
        return mido.MidiFile(str(midi_path))
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def midi_to_pretty_midi(midi_obj: mido.MidiFile) -> pretty_midi.PrettyMIDI:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        midi_obj.save(str(tmp_path))
        return pretty_midi.PrettyMIDI(str(tmp_path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def offset_pretty_midi(pm: pretty_midi.PrettyMIDI, offset_seconds: float) -> pretty_midi.PrettyMIDI:
    shifted = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for instrument in pm.instruments:
        dst = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
        for note in instrument.notes:
            dst.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start + offset_seconds,
                    end=note.end + offset_seconds,
                )
            )
        shifted.instruments.append(dst)
    return shifted


def deduplicate_notes(instrument: pretty_midi.Instrument, onset_tol: float, offset_tol: float) -> None:
    instrument.notes.sort(key=lambda n: (n.pitch, n.start, n.end, n.velocity))
    kept = []
    for note in instrument.notes:
        duplicate = False
        for prev in reversed(kept):
            if prev.pitch != note.pitch:
                break
            if abs(prev.start - note.start) <= onset_tol and (
                instrument.is_drum or abs(prev.end - note.end) <= offset_tol
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(note)
    instrument.notes = kept


def merge_segment_midis(
    segments: Iterable[tuple[float, mido.MidiFile]],
    onset_tol: float,
    offset_tol: float,
    note_time_shift: float,
) -> pretty_midi.PrettyMIDI:
    grouped = {}
    for offset_seconds, midi_obj in segments:
        shifted = offset_pretty_midi(midi_to_pretty_midi(midi_obj), offset_seconds + note_time_shift)
        for instrument in shifted.instruments:
            key = (instrument.program, instrument.is_drum, instrument.name or "")
            if key not in grouped:
                grouped[key] = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
            grouped[key].notes.extend(instrument.notes)
    merged = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for instrument in grouped.values():
        for note in instrument.notes:
            if note.end <= 0:
                continue
            note.start = max(0.0, note.start)
            if note.end <= note.start:
                note.end = note.start + 0.02
        deduplicate_notes(instrument, onset_tol=onset_tol, offset_tol=offset_tol)
        merged.instruments.append(instrument)
    return merged


@torch.inference_mode()
def transcribe_file(
    model: YourMT3,
    stem: Path,
    out_mid: Path,
    device: str,
    chunk_seconds: float,
    overlap_seconds: float,
    dedupe_onset_tol: float,
    dedupe_offset_tol: float,
    note_time_shift: float,
    infer_bsz: int,
) -> float:
    t0 = time.time()
    audio, sr = librosa.load(stem, sr=16000, mono=True)
    pieces = split_audio(audio.astype("float32"), sr, chunk_seconds, overlap_seconds)
    transcribed = []
    work_root = CACHE_ROOT / "tmp"
    for offset, chunk in pieces:
        features = preprocess_audio(model, chunk, sr).to(device)
        outputs, _loss = model.inference_file(bsz=infer_bsz, audio_segments=features)
        transcribed.append((offset, decode_outputs_to_mido(model, outputs, work_root)))
    merged = merge_segment_midis(
        transcribed,
        onset_tol=dedupe_onset_tol,
        offset_tol=dedupe_offset_tol,
        note_time_shift=note_time_shift,
    )
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    merged.write(str(out_mid))
    return time.time() - t0
