# Model card: `sub_saharan`

> Scaffolded by `prep_zenodo_staging.py` 2026-05-22. Sections marked
> `[FILL IN]` need upstream-specific content before public release.

## Model kind

Detector (YOLO end-to-end with embedded NMS)

## Intended use

[FILL IN — single sentence describing what this model is for, e.g. "MegaDetector v6 detects animals, humans, and vehicles in camera-trap images."]

## Training data summary

[FILL IN — dataset name, approximate size, collection methodology, geographic / taxonomic scope]

## Performance numbers

[FILL IN — published mAP / precision / recall on a benchmark, or "no published numbers" if not available. Cite the benchmark.]

## Input shape + preprocessing

(Auto-extracted from `manifest.toml`; confirm correctness.)

- Input size: `[640, 480]`
- Layout: `nchw`
- Preprocessing method: `letterbox`
- Normalization: `unit`

## Output shape

[FILL IN — output tensor shape + class count. For detectors: bbox + score + class dims. For classifiers: vector length = number of classes.]

## Upstream source

- URL: [FILL IN — original PyTorch/TF model release URL]
- Version / commit / release tag: [FILL IN]

## Conversion notes

[FILL IN — describe the upstream → ONNX conversion path. If converted via a script, include the script name + sparrow-engine commit at conversion time. If upstream ships ONNX directly, say "Used upstream ONNX directly; no conversion."]

## BibTeX citation

```
[FILL IN — BibTeX entry for the upstream paper. Replace this block with the actual entry.]
```

## License

See `LICENSE.md` in this directory.

## Usage with sparrow-engine

```bash
spe detect --model sub_saharan --image /path/to/image.jpg
```

See the bundle-level `README.md` for installation + general usage.
