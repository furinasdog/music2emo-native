"""Export the three Music2Emo inference graphs. Python is needed only for conversion."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


class EmotionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.chord_root_embedding = nn.Embedding(14, 4)
        self.chord_attr_embedding = nn.Embedding(14, 4)
        positions = torch.arange(100).float().unsqueeze(1)
        angles = positions * torch.exp(torch.arange(0, 8, 2).float() * (-np.log(10000.0) / 8))
        encoding = torch.zeros(1, 100, 8)
        encoding[0, :, 0::2] = angles.sin()
        encoding[0, :, 1::2] = angles.cos()
        self.register_buffer("encoding", encoding, persistent=False)
        self.chord_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(8, 8, dim_feedforward=64, batch_first=True),
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.input_proj = nn.Sequential(nn.Linear(1545, 512), nn.ReLU())
        self.classification_branch = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 56))
        self.regression_branch = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, embedding, roots, attributes, minor):
        parts = (self.chord_root_embedding(roots), self.chord_attr_embedding(attributes))
        harmony = torch.cat(parts, dim=-1) + self.encoding
        harmony = torch.cat((torch.zeros_like(harmony[:, :1]), harmony), dim=1)
        pooled = self.chord_transformer(harmony)[:, 0, :]
        layer = self.input_proj(torch.cat((embedding, pooled, minor.float()), dim=-1))
        return self.classification_branch(layer), self.regression_branch(layer)


class AcousticEmbedding(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        states = self.model(waveform, output_hidden_states=True).hidden_states
        return torch.cat((states[6].mean(dim=1), states[7].mean(dim=1)), dim=-1)


class ChordPrediction(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        encoded, _ = self.model.self_attn_layers(features)
        classes, _ = self.model.output_layer(encoded)
        return classes.reshape(1, 108).long()


def state_dict(path: Path, key: str) -> tuple[dict, dict]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as archive:
            return {name: archive.get_tensor(name) for name in archive.keys()}, archive.metadata() or {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint[key], checkpoint


def export(module, values, target, inputs, outputs, axes=None):
    import onnxruntime as ort

    module.eval()
    with torch.no_grad():
        reference = module(*values)
        torch.onnx.export(
            module, values, target, opset_version=17, dynamo=False,
            input_names=inputs, output_names=outputs, dynamic_axes=axes,
        )
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    results = session.run(outputs, dict(zip(inputs, (value.numpy() for value in values))))
    reference = reference if isinstance(reference, tuple) else (reference,)
    for result, expected in zip(results, reference):
        np.testing.assert_allclose(result, expected.detach().numpy(), rtol=1e-3, atol=2e-4)
    print(f"Verified {target.name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--mert", type=Path, required=True)
    parser.add_argument("--btc-checkpoint", type=Path)
    parser.add_argument("--emotion-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["btc", "emotion", "mert", "all"], default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(17)
    data = args.upstream / "inference" / "data"
    if args.stage in ("mert", "all"):
        from transformers import AutoModel

        model = AutoModel.from_pretrained(str(args.mert), trust_remote_code=True, local_files_only=True)
        model.encoder.layers = nn.ModuleList(list(model.encoder.layers[:7]))
        export(AcousticEmbedding(model), (torch.randn(1, 48000),), args.output / "mert.onnx",
               ["waveform"], ["embedding"], {"waveform": {1: "samples"}})
    if args.stage in ("btc", "all"):
        import yaml

        sys.path.insert(0, str(args.upstream.resolve()))
        from utils import btc_model

        settings = yaml.safe_load((data / "run_config.yaml").read_text(encoding="utf-8"))
        weights, metadata = state_dict(args.btc_checkpoint or data / "btc_model_large_voca.pt", "model")
        btc_model.use_cuda = False
        btc = btc_model.BTC_model(config=settings["model"])
        btc.load_state_dict(weights)
        export(ChordPrediction(btc), (torch.randn(1, 108, 144),), args.output / "btc.onnx", ["features"], ["chords"])
        attributes = json.loads((data / "chord_attr.json").read_text(encoding="utf-8"))
        order = ("min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4")
        tags = np.load(data / "tag_list.npy")[127:].tolist()
        if len(tags) != 56:
            raise ValueError("Unexpected Music2Emo tag count")
        (args.output / "metadata.txt").write_text(
            "Music2Emo-ONNX-1\n" + f"{float(metadata['mean'])} {float(metadata['std'])}\n"
            + " ".join(str(attributes.get(name, attributes["min7"])) for name in order) + "\n"
            + "\n".join(tag.removeprefix("mood/theme---") for tag in tags) + "\n",
            encoding="utf-8", newline="\n",
        )
    if args.stage in ("emotion", "all"):
        model = EmotionHead()
        weights, _ = state_dict(args.emotion_checkpoint or args.upstream / "saved_models" / "J_all.ckpt", "state_dict")
        weights = {name.removeprefix("model."): tensor for name, tensor in weights.items()}
        model.load_state_dict({name: weights[name] for name in model.state_dict()})
        values = (torch.randn(1, 1536), torch.randint(0, 13, (1, 100)), torch.randint(0, 14, (1, 100)), torch.ones(1, 1, dtype=torch.long))
        export(model, values, args.output / "emotion.onnx", ["embedding", "roots", "attributes", "minor"], ["logits", "valence_arousal"])


if __name__ == "__main__":
    main()
