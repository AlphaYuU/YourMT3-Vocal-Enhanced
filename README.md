A MIDI transcription model focuses on vocal, fine-tuned on the MIR-ST500 Chinese vocal.

## Files

- `model.pt` - inference weights.
- `config.json` - model/audio/task configuration.
- `r47b_infer.py` - local vocal stem to MIDI CLI.
- `runtime_deps/`, `yourmt3_train_src/` - minimal runtime source.

## Usage

```bash
pip install -r requirements.txt
python r47b_infer.py vocals.wav --output-midi vocals.mid --device cuda
```

Input should be a separated vocal stem.

## Metrics

Metrics are reported on MIR-ST500 vocal transcription splits; this model repo
does not include dataset audio or labels.

| Route | Split | COnP-F1 | COn-F1 | COnPOff-F1 |
|---|---|---:|---:|---:|
| r47b direct `model.pt` | official82 | `0.6882375797` | `0.6882375797` | `0.5143113621` |
| r47b + E12 single-source | official82 | `0.7181716362` | `0.7624981992` | `0.5220578867` |
| r47b + E12 single-source | train337/calib | `0.7468827545` | `0.7744380415` | `0.5575166125` |
