# Music2Emo Native

Local, segment-level [Music2Emo](https://github.com/AMAAI-Lab/Music2Emotion) inference for Revelation. The executable uses ONNX Runtime with the CPU Execution Provider, including for MERT, BTC chord recognition, and the Music2Emo emotion head. No Python process is needed at runtime.

## Build

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release "-DONNXRUNTIME_ROOT=C:/sdk/onnxruntime-1.23.2"
cmake --build build
```

The executable accepts raw little-endian float32 mono audio at 24 kHz, an output JSON path, and the directory containing the four model files. FFmpeg is one way to prepare the input.

```powershell
ffmpeg -i song.wav -ac 1 -ar 24000 -f f32le song.f32
./build/revelation-music2emo song.f32 emotions.json ./models
```

The output contains non-overlapping 30-second windows with original audio timestamps. Each window has up to three ranked mood/theme tags, with independent sigmoid scores, and valence and arousal on a 1–9 scale. The final window ends at the audio duration. A label is an estimate about the music.
An optional fourth argument is a progress file. The process writes the number of finished windows and the total, separated by a space.

## Model files

The runtime requires `mert.onnx`, `btc.onnx`, `emotion.onnx`, and `metadata.txt` in the same directory. In the editor, choose 「情绪分析 → 下载模型」 to install the [published ONNX bundle](https://github.com/furinasdog/music2emo-native/releases/tag/models-v1). To install manually, extract the bundle and select its directory in the editor. The editor can also discover the directory from the `AUTOREVELATION_MUSIC2EMO_MODELS` environment variable.

Use the [Music2Emo repository](https://github.com/AMAAI-Lab/Music2Emotion/tree/c5f7984f78d9928b841f25bd05a47377c12c58ca) and the [MERT checkpoint](https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5) to prepare the assets:

```powershell
python -m pip install torch 'transformers==4.46.3' onnx onnxruntime numpy pyyaml safetensors
python conversion/export_models.py --upstream ./Music2Emotion --mert ./MERT-v1-95M --output ./models
```

The exporter can also take the repacked tensor-only `J_all.safetensors` and `btc_model_large_voca.safetensors` from the [Music2Emo inference assets repack](https://huggingface.co/mementoor/music2emo-inference-assets), using `--emotion-checkpoint` and `--btc-checkpoint`. Music2Emo code is MIT and the [emotion checkpoint package](https://huggingface.co/amaai-lab/music2emo/tree/b036e59471583c3d5b30c69e63e8c7323cc36c4a) is Apache-2.0. MERT weights and ONNX derivatives are [CC BY-NC 4.0](https://huggingface.co/m-a-p/MERT-v1-95M) and supplied separately from the editor. Keep the corresponding attribution and license with redistributed model files.
The tested MERT repack is `mert/model.safetensors.zst` with SHA-256 `6f8e05b485d31606b451e5897892db2e5550732b81516e7fda28a2eaf68dd22b`. Decompress it to `model.safetensors` next to the MERT config and Python definitions and set `num_hidden_layers` to `7` in that config. The ONNX export uses the sixth and seventh encoder layers.

## Inference differences

The neural graphs use the upstream learned weights. The port computes 144-bin constant-Q spectra with Hann windows at fixed tuning, estimates key from predicted chord durations, and applies the emotion model separately for each editing window. The reference Python application uses librosa tuning/resampling and reads the song's key via music21. These differences can change the scores, particularly around short sections and modulations. `export_models.py` checks PyTorch versus ONNX output for each exported graph; this checks graph conversion rather than end-to-end equivalence of the front ends.

## Citation

Jaeyong Kang and Dorien Herremans, [Towards Unified Music Emotion Recognition across Dimensional and Categorical Models](https://arxiv.org/abs/2502.03979).
