"""Extract speech model features for entire audio files specified in an item file.

This script extracts features for the entire audio files referenced in the ABX item file.
It saves all features for each layer into a single consolidated file.

Output:
  - For neural models: {output_dir}/{dataset_name}/{model_name}/l{i}/features.pt
  - For MFCC/FBANK: {output_dir}/{dataset_name}/{model_name}/features.pt

Each features.pt contains:
  - "features": {file_id: tensor, ...}
  - "frequency": float (50.0 for SSL, 100.0 for MFCC/FBANK)

Usage:
    python extract_features.py abx_items/pitch_accent features --model HUBERT_BASE
    # Results will be in features/pitch_accent/hubert_base/l{1..12}/features.pt

    python extract_features.py abx_items/pitch_accent features --model MFCC
    # Results will be in features/pitch_accent/mfcc/features.pt
"""

import argparse
import csv
import shutil
from pathlib import Path

import torch
import torchaudio
from torch.nn import functional as F
from tqdm import tqdm
from torchaudio.compliance.kaldi import fbank, mfcc


# HuBERT/WavLM have 20ms frame shift = 50 Hz
SSL_FREQUENCY = 50.0
# Kaldi MFCC/FBANK use 10ms frame shift = 100 Hz
KALDI_FREQUENCY = 100.0
MFCC_SAMPLE_RATE = 16_000
HF_MODELS = {
    # English
    "wav2vec2-large-xlsr-53-english": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    # Chinese
    "chinese_hubert_large": "TencentGameMate/chinese-hubert-large",
    "chinese_hubert_base": "TencentGameMate/chinese-hubert-base",
    "chinese_wav2vec2_base": "TencentGameMate/chinese-wav2vec2-base",
    "chinese_wav2vec2_large": "TencentGameMate/chinese-wav2vec2-large",
    "wav2vec2-large-xlsr-53-chinese-zh-cn": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    # Japanese
    "japanese-wav2vec2-base": "yky-h/japanese-wav2vec2-base",
    "japanese-hubert-base": "yky-h/japanese-hubert-base",
    "japanese-hubert-large": "yky-h/japanese-hubert-large",
    "wav2vec2-large-xlsr-53-japanese": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    # Multilingual
    "w2v-bert-2.0": "facebook/w2v-bert-2.0",
    "mhubert": "utter-project/mHuBERT-147",
}


def extract_features(
    item_file: Path,
    audio_root: Path,
    output_root: Path,
    model_name: str,
    extension: str = ".wav",
    low_memory: bool = False,
):
    """Extract features for entire files specified in an item file.

    Args:
        item_file: Path to item CSV file with #file column
        audio_root: Directory containing audio files
        output_root: Directory to save features
        model_name: Model name (e.g., HUBERT_BASE, WAVLM_BASE_PLUS, MFCC)
        extension: Audio file extension
        low_memory: Use two-phase extraction to reduce peak memory (slower)
    """
    model_key_upper = model_name.upper()
    model_key_lower = model_name.lower()

    use_mfcc = model_key_upper == "MFCC"
    use_fbank = model_key_upper == "FBANK"
    use_kaldi = use_mfcc or use_fbank

    # Check if we can skip entirely
    if use_kaldi:
        output_path = output_root / "features.pt"
    else:
        output_path = output_root / "l1" / "features.pt"

    if output_path.exists():
        print(f"Skipping: {output_path} already exists")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_hf = False
    hf_model = None
    hf_processor = None

    if use_kaldi:
        if use_mfcc:
            print("Using MFCC features (torchaudio.compliance.kaldi).")
        else:
            print("Using FBANK features (torchaudio.compliance.kaldi).")
        model = None
        expected_sr = MFCC_SAMPLE_RATE
        frequency = KALDI_FREQUENCY
    else:
        # Load model
        print(f"Loading {model_name}...")
        bundle = getattr(torchaudio.pipelines, model_key_upper, None)
        if bundle is not None:
            model = bundle.get_model().eval().to(device)
            expected_sr = bundle.sample_rate
        elif model_key_lower in HF_MODELS:
            from transformers import AutoFeatureExtractor, AutoModel
            hf_name = HF_MODELS[model_key_lower]
            hf_processor = AutoFeatureExtractor.from_pretrained(hf_name)
            hf_model = AutoModel.from_pretrained(hf_name).to(device).eval()
            expected_sr = getattr(hf_processor, "sampling_rate", MFCC_SAMPLE_RATE)
            use_hf = True
            model = None
        else:
            raise ValueError(f"Unknown model: {model_name}")
        frequency = SSL_FREQUENCY

    # Read item file and find unique files
    print(f"Reading item file: {item_file}")
    with open(item_file) as f:
        reader = csv.DictReader(f)
        items = list(reader)

    # Get unique file IDs
    file_ids = sorted(list(set(item["#file"] for item in items)))
    print(f"Found {len(file_ids)} unique files to process.")

    output_root.mkdir(exist_ok=True, parents=True)

    if use_kaldi:
        # For Kaldi (MFCC/FBANK): accumulate in memory (features are small)
        kaldi_features: dict[str, torch.Tensor] = {}

        for file_id in tqdm(file_ids, desc="Extracting features"):
            audio_path = audio_root / f"{file_id}{extension}"
            if not audio_path.exists():
                raise FileNotFoundError(f"{audio_path} not found")

            try:
                audio, sample_rate = torchaudio.load(str(audio_path))
            except Exception as e:
                raise RuntimeError(f"Error loading {audio_path}: {e}") from e

            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            if sample_rate != expected_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, expected_sr).to(audio.device)
                audio = resampler(audio)

            with torch.no_grad():
                if use_mfcc:
                    m = mfcc(audio)
                else:
                    m = fbank(audio)
            if m.dim() == 3 and m.shape[0] == 1:
                m = m.squeeze(0)
            kaldi_features[file_id] = m.cpu()

        output_path = output_root / "features.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": kaldi_features, "frequency": frequency}, output_path)
        print(f"Saved {len(kaldi_features)} files to {output_path}")

    else:
        # For neural models
        layer_features: dict[int, dict[str, torch.Tensor]] = {}
        num_layers = None

        if low_memory:
            # Two-phase extraction: save per-file to temp, then consolidate layer-by-layer
            # Reduces peak memory but reads temp files num_layers times
            import os
            local_user_dir = Path(f"/localdisk/{os.environ.get('USER', 'tmp')}")
            temp_base = local_user_dir / "fastabx_temp"
            # Use deterministic temp dir name based on dataset/model for resumability
            temp_dir = temp_base / f"{output_root.parent.name}_{output_root.name}"
            temp_dir.mkdir(exist_ok=True, parents=True)
            print(f"Using low-memory mode (temp dir: {temp_dir})")

            # Phase 1: Extract and save per-file
            print("Phase 1: Extracting features per file...")
            for i, file_id in enumerate(tqdm(file_ids, desc="Extracting features")):
                temp_path = temp_dir / f"{i}.pt"
                if temp_path.exists():
                    if num_layers is None:
                        _, data = torch.load(temp_path, weights_only=False)
                        num_layers = len(data)
                    continue

                audio_path = audio_root / f"{file_id}{extension}"
                if not audio_path.exists():
                    raise FileNotFoundError(f"{audio_path} not found")

                try:
                    audio, sample_rate = torchaudio.load(str(audio_path))
                except Exception as e:
                    raise RuntimeError(f"Error loading {audio_path}: {e}") from e

                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)

                if sample_rate != expected_sr:
                    resampler = torchaudio.transforms.Resample(sample_rate, expected_sr).to(audio.device)
                    audio = resampler(audio)

                if use_hf:
                    with torch.no_grad():
                        inputs = hf_processor(
                            audio.squeeze(0),
                            sampling_rate=expected_sr,
                            return_tensors="pt",
                        ).to(device)
                        out = hf_model(**inputs, output_hidden_states=True)
                    features = [h[0].cpu() for h in out.hidden_states[1:]]
                else:
                    audio = F.layer_norm(audio, audio.shape)
                    with torch.no_grad():
                        features, _ = model.extract_features(audio.to(device))
                    features = [f.squeeze(0).cpu() for f in features]

                if num_layers is None:
                    num_layers = len(features)

                torch.save((file_id, features), temp_path)

            # Phase 2: Consolidate layer-by-layer
            print("Phase 2: Consolidating features by layer...")
            for layer_idx in range(1, num_layers + 1):
                layer_dir = output_root / f"l{layer_idx}"
                output_path = layer_dir / "features.pt"

                if output_path.exists():
                    print(f"Skipping layer {layer_idx}: {output_path} already exists")
                    continue

                print(f"Consolidating layer {layer_idx}...")
                layer_features_single = {}
                for temp_file in tqdm(sorted(temp_dir.glob("*.pt")), desc=f"Loading layer {layer_idx}"):
                    file_id, data = torch.load(temp_file, weights_only=False)
                    layer_features_single[file_id] = data[layer_idx - 1]

                layer_dir.mkdir(exist_ok=True)
                torch.save({"features": layer_features_single, "frequency": frequency}, output_path)
                print(f"Saved layer {layer_idx} ({len(layer_features_single)} files) to {output_path}")
                del layer_features_single

            print("Cleaning up temporary files...")
            shutil.rmtree(temp_dir)

        else:
            # Default: accumulate all layers in memory (faster, uses more RAM)
            for file_id in tqdm(file_ids, desc="Extracting features"):
                audio_path = audio_root / f"{file_id}{extension}"
                if not audio_path.exists():
                    raise FileNotFoundError(f"{audio_path} not found")

                try:
                    audio, sample_rate = torchaudio.load(str(audio_path))
                except Exception as e:
                    raise RuntimeError(f"Error loading {audio_path}: {e}") from e

                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)

                if sample_rate != expected_sr:
                    resampler = torchaudio.transforms.Resample(sample_rate, expected_sr).to(audio.device)
                    audio = resampler(audio)

                if use_hf:
                    with torch.no_grad():
                        inputs = hf_processor(
                            audio.squeeze(0),
                            sampling_rate=expected_sr,
                            return_tensors="pt",
                        ).to(device)
                        out = hf_model(**inputs, output_hidden_states=True)
                    features = [h[0].cpu() for h in out.hidden_states[1:]]
                else:
                    audio = F.layer_norm(audio, audio.shape)
                    with torch.no_grad():
                        features, _ = model.extract_features(audio.to(device))
                    features = [f.squeeze(0).cpu() for f in features]

                if num_layers is None:
                    num_layers = len(features)
                    for i in range(num_layers):
                        layer_features[i] = {}

                for i, feat in enumerate(features):
                    layer_features[i][file_id] = feat

            # Save all layers
            for i in range(num_layers):
                layer_idx = i + 1
                layer_dir = output_root / f"l{layer_idx}"
                output_path = layer_dir / "features.pt"

                if output_path.exists():
                    print(f"Skipping layer {layer_idx}: {output_path} already exists")
                    continue

                layer_dir.mkdir(exist_ok=True)
                torch.save({"features": layer_features[i], "frequency": frequency}, output_path)
                print(f"Saved layer {layer_idx} ({len(layer_features[i])} files) to {output_path}")

    print(f"Features saved to {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract speech model features for entire files in an item file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("dataset_path", type=Path, help="Directory containing items.csv and audio_path.txt")
    parser.add_argument("output", type=Path, help="Top-level directory to save features (dataset/model subdirectories will be created)")

    parser.add_argument(
        "--model",
        type=str,
        default="HUBERT_BASE",
        help="Model to use for feature extraction (torchaudio pipeline, HF model, MFCC, or FBANK)",
    )
    parser.add_argument("--extension", type=str, default=".wav", help="Audio file extension")
    parser.add_argument("--low-memory", action="store_true",
                        help="Use two-phase extraction to reduce peak memory (slower, reads temp files N times)")

    args = parser.parse_args()

    # Resolve paths
    item_file = args.dataset_path / "items.csv"
    if not item_file.exists():
        raise FileNotFoundError(f"Item file not found: {item_file}")

    audio_path_file = args.dataset_path / "audio_path.txt"
    if not audio_path_file.exists():
        raise FileNotFoundError(
            f"Audio path file not found: {audio_path_file}. "
            "Please ensure the dataset generation script created it or provide the path manually."
        )

    with open(audio_path_file, "r") as f:
        audio_root = Path(f.read().strip())

    # Resolve dataset name (e.g. 'pitch_accent' or 'ap')
    dataset_name = args.dataset_path.name

    # Create output directory with dataset and model name
    # special handling for CSJ_unit to just use 'csj'
    if dataset_name.startswith("csj_"):
        dataset_name = "csj"
    output_root = args.output / dataset_name / args.model.lower()

    print(f"Dataset directory: {args.dataset_path}")
    print(f"Audio root (from audio_path.txt): {audio_root}")
    print(f"Output directory: {output_root}")

    extract_features(
        item_file=item_file,
        audio_root=audio_root,
        output_root=output_root,
        model_name=args.model,
        extension=args.extension,
        low_memory=args.low_memory,
    )


if __name__ == "__main__":
    main()
