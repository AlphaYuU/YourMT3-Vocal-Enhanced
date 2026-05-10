"""Run the r47b single-source vocal-to-MIDI model export.

This standard model-repo CLI runs direct inference from ``model.pt`` and
``config.json``. The full arbitrary-audio E12 pitch-vote runtime is not part of
this lightweight model release.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "model.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"
OFFICIAL82_DIRECT_CONP_F1 = 0.6882375796712107


def _count_notes(midi_path: Path) -> int | None:
    try:
        import pretty_midi

        return sum(len(inst.notes) for inst in pretty_midi.PrettyMIDI(str(midi_path)).instruments)
    except Exception:
        return None


def _write_diagnostics(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_direct_model(
    *,
    model_path: Path,
    config_path: Path,
    input_audio: Path,
    output_midi: Path,
    cache_dir: Path,
    device: str,
    chunk_seconds: float,
    overlap_seconds: float,
    infer_bsz: int,
) -> dict[str, Any]:
    if not input_audio.exists():
        raise FileNotFoundError(input_audio)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    os.environ.setdefault("R47B_PROJECT_ROOT", str(PROJECT_ROOT))
    os.environ["R47B_CACHE_ROOT"] = str(cache_dir.resolve())

    from runtime_deps import local_checkpoint

    model = local_checkpoint.load_local_model(model_path, config_path, device)
    try:
        elapsed_s = local_checkpoint.transcribe_file(
            model,
            input_audio,
            output_midi,
            device=device,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            dedupe_onset_tol=0.03,
            dedupe_offset_tol=0.05,
            note_time_shift=0.0,
            infer_bsz=infer_bsz,
        )
    finally:
        try:
            import torch

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "route": "r47b_direct_model_pt",
        "input_audio": str(input_audio),
        "model": str(model_path),
        "config": str(config_path),
        "output_midi": str(output_midi),
        "note_count": _count_notes(output_midi),
        "elapsed_s": elapsed_s,
        "metrics_scope": "arbitrary audio direct model.pt inference; not E12 pitch-vote",
        "official82_direct_conp_f1": OFFICIAL82_DIRECT_CONP_F1,
        "release_limitation": "Full arbitrary-audio E12 pitch-vote is not included in this standard model repo.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_audio", type=Path, help="Mono vocal stem audio path.")
    parser.add_argument("--output-midi", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "r47b")
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-seconds", type=float, default=12.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--infer-bsz", type=int, default=1)
    args = parser.parse_args(argv)

    output_midi = args.output_midi.resolve()
    diag = _run_direct_model(
        model_path=args.model.resolve(),
        config_path=args.config.resolve(),
        input_audio=args.input_audio.resolve(),
        output_midi=output_midi,
        cache_dir=args.cache_dir,
        device=args.device,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        infer_bsz=args.infer_bsz,
    )

    _write_diagnostics(args.diagnostics.resolve() if args.diagnostics else None, diag)
    print(f"[r47b single-source] route={diag['route']} output={output_midi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
