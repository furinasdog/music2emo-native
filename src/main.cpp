#include "audio_features.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
namespace fs = std::filesystem;
using namespace music2emo;
constexpr int tags_count = 56;
constexpr int embedding_size = 1536;
constexpr int window_samples = 30 * input_rate;
constexpr int low_memory_chunk_samples = 6 * input_rate;
constexpr int maximum_samples = 900 * input_rate;

struct Metadata {
    float mean{}, scale{};
    std::array<std::int64_t, 14> attributes{};
    std::array<std::string, tags_count> tags;
};

Metadata load_metadata(const fs::path& root) {
    std::ifstream stream(root / "metadata.txt");
    stream.imbue(std::locale::classic());
    std::string revision;
    std::getline(stream, revision);
    if (revision != "Music2Emo-ONNX-1" || !(stream >> std::ws)) throw std::runtime_error("Invalid Music2Emo metadata");
    Metadata metadata;
    stream >> metadata.mean >> metadata.scale;
    if (!std::isfinite(metadata.mean) || !std::isfinite(metadata.scale) || metadata.scale <= 0.0f)
        throw std::runtime_error("Invalid BTC normalization");
    for (auto& index : metadata.attributes) {
        if (!(stream >> index) || index < 0 || index >= 14) throw std::runtime_error("Invalid BTC chord mapping");
    }
    stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    for (auto& label : metadata.tags) {
        if (!std::getline(stream, label) || label.empty()
            || label.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_") != std::string::npos)
            throw std::runtime_error("Invalid Music2Emo tag");
    }
    return metadata;
}

std::vector<float> read_pcm(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto bytes = stream.tellg();
    if (bytes <= 0 || bytes % sizeof(float) != 0 || bytes > std::streamoff(maximum_samples) * sizeof(float))
        throw std::runtime_error("Expected at most 15 minutes of mono float32 PCM at 24 kHz");
    std::vector<float> audio(static_cast<std::size_t>(bytes) / sizeof(float));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(audio.data()), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("Incomplete audio data");
    for (float value : audio) {
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite audio sample");
    }
    return audio;
}

bool low_memory_required() {
#ifdef _WIN32
    constexpr auto threshold = 3ULL * 1024 * 1024 * 1024;
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory) && (memory.ullAvailPhys < threshold || memory.ullAvailPageFile < threshold))
        return true;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION job{};
    if (QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &job, sizeof(job), nullptr)) {
        const auto flags = job.BasicLimitInformation.LimitFlags;
        if ((flags & JOB_OBJECT_LIMIT_JOB_MEMORY) && job.JobMemoryLimit < threshold) return true;
        if ((flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) && job.ProcessMemoryLimit < threshold) return true;
    }
#endif
    return false;
}

Ort::SessionOptions create_options(bool low_memory) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(static_cast<int>(std::clamp(std::thread::hardware_concurrency() / 2, 1u, 4u)));
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (low_memory) {
        options.DisableCpuMemArena();
        options.DisableMemPattern();
    }
    return options;
}

template <typename T, std::size_t N>
Ort::Value tensor(const Ort::MemoryInfo& memory, T* input, std::size_t count, const std::array<std::int64_t, N>& shape) {
    return Ort::Value::CreateTensor<T>(memory, input, count, shape.data(), shape.size());
}

class Pipeline {
public:
    Pipeline(const fs::path& root, bool low_memory)
        : metadata_(load_metadata(root)), low_memory_(low_memory), options_(create_options(low_memory_)),
          mert_(environment_, (root / "mert.onnx").c_str(), options_),
          btc_(environment_, (root / "btc.onnx").c_str(), options_),
          head_(environment_, (root / "emotion.onnx").c_str(), options_) {}

    void infer(const std::vector<float>& audio, std::ostream& output, const fs::path& progress_path) {
        const int total = (static_cast<int>(audio.size()) + window_samples - 1) / window_samples;
        report_progress(progress_path, 0, total);
        output << "{\"model\":\"Music2Emo\",\"window_ms\":30000,\"low_memory\":"
               << (low_memory_ ? "true" : "false") << ",\"segments\":[";
        for (int start = 0; start < static_cast<int>(audio.size()); start += window_samples) {
            const int length = std::min(window_samples, static_cast<int>(audio.size()) - start);
            write_window(audio.data() + start, start, length, output);
            report_progress(progress_path, start / window_samples + 1, total);
        }
        output << "]}";
    }

private:
    Metadata metadata_;
    ConstantQ constant_q_;
    Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "Music2Emo"};
    bool low_memory_;
    Ort::SessionOptions options_;
    Ort::Session mert_, btc_, head_;
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    static void report_progress(const fs::path& path, int complete, int total) {
        if (path.empty()) return;
        std::ofstream progress(path, std::ios::binary);
        progress << complete << ' ' << total << '\n';
    }

    std::array<float, embedding_size> acoustic_embedding(const float* audio, int length) {
        std::array<float, embedding_size> embedding{};
        constexpr std::array mert_input{"waveform"};
        constexpr std::array mert_output{"embedding"};
        const int chunk_size = low_memory_ ? low_memory_chunk_samples : length;
        // The MERT graph pools each chunk in time. Weight those embeddings by
        // their actual audio lengths so a short final chunk stays proportional.
        for (int offset = 0; offset < length; offset += chunk_size) {
            const int count = std::min(chunk_size, length - offset);
            auto waveform = normalize_waveform(audio + offset, count, input_rate * 2);
            const std::array<std::int64_t, 2> shape{1, static_cast<std::int64_t>(waveform.size())};
            auto input = tensor(memory_, waveform.data(), waveform.size(), shape);
            auto extracted = mert_.Run(Ort::RunOptions{nullptr}, mert_input.data(), &input, 1, mert_output.data(), 1);
            if (extracted[0].GetTensorTypeAndShapeInfo().GetElementCount() != embedding_size)
                throw std::runtime_error("Incorrect MERT output");
            const auto* values = extracted[0].GetTensorData<float>();
            const float weight = static_cast<float>(count) / static_cast<float>(length);
            for (std::size_t index = 0; index < embedding.size(); ++index)
                embedding[index] += values[index] * weight;
        }
        return embedding;
    }

    void write_window(const float* audio, int start, int length, std::ostream& output) {
        auto embedding = acoustic_embedding(audio, length);
        auto chords = resample_for_chords(audio, length);
        std::vector<int> classes;
        for (int offset = 0; offset < static_cast<int>(chords.size()); offset += 10 * chord_rate) {
            const int samples = std::min(10 * chord_rate, static_cast<int>(chords.size()) - offset);
            auto features = constant_q_.transform(chords.data() + offset, samples, metadata_.mean, metadata_.scale);
            const auto count = std::min<std::size_t>(features.size() / bins, chord_frames);
            features.resize(chord_frames * bins, 0.0f);
            constexpr std::array<std::int64_t, 3> shape{1, chord_frames, bins};
            auto input = tensor(memory_, features.data(), features.size(), shape);
            constexpr std::array input_names{"features"};
            constexpr std::array output_names{"chords"};
            const auto result = btc_.Run(Ort::RunOptions{nullptr}, input_names.data(), &input, 1, output_names.data(), 1);
            if (result[0].GetTensorTypeAndShapeInfo().GetElementCount() != chord_frames) throw std::runtime_error("Incorrect BTC output");
            const auto* predictions = result[0].GetTensorData<std::int64_t>();
            for (std::size_t i = 0; i < count; ++i) {
                if (predictions[i] < 0 || predictions[i] > 169) throw std::runtime_error("Invalid BTC chord");
                classes.push_back(static_cast<int>(predictions[i]));
            }
        }
        auto harmony = encode_harmony(classes, metadata_.attributes);
        constexpr std::array<std::int64_t, 2> embedding_shape{1, embedding_size}, chord_shape{1, max_chords}, key_shape{1, 1};
        std::array inputs{
            tensor(memory_, embedding.data(), embedding.size(), embedding_shape),
            tensor(memory_, harmony.roots.data(), harmony.roots.size(), chord_shape),
            tensor(memory_, harmony.attributes.data(), harmony.attributes.size(), chord_shape),
            tensor(memory_, &harmony.minor, 1, key_shape),
        };
        constexpr std::array input_names{"embedding", "roots", "attributes", "minor"};
        constexpr std::array output_names{"logits", "valence_arousal"};
        const auto result = head_.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                                      output_names.data(), output_names.size());
        if (result[0].GetTensorTypeAndShapeInfo().GetElementCount() != tags_count
            || result[1].GetTensorTypeAndShapeInfo().GetElementCount() != 2)
            throw std::runtime_error("Incorrect Music2Emo checkpoint outputs");
        const auto* scores = result[0].GetTensorData<float>();
        const auto* coordinates = result[1].GetTensorData<float>();
        if (!std::isfinite(coordinates[0]) || !std::isfinite(coordinates[1])) throw std::runtime_error("Non-finite emotion prediction");
        std::array<int, tags_count> order{};
        std::iota(order.begin(), order.end(), 0);
        for (int index : order) {
            if (!std::isfinite(scores[index])) throw std::runtime_error("Non-finite tag prediction");
        }
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return scores[x] > scores[y]; });
        output << (start == 0 ? "" : ",") << "{\"time_ms\":" << start / 24 << ",\"end_ms\":" << (start + length + 23) / 24
               << ",\"valence\":" << std::clamp(coordinates[0], 1.0f, 9.0f)
               << ",\"arousal\":" << std::clamp(coordinates[1], 1.0f, 9.0f) << ",\"tags\":[";
        for (int rank = 0; rank < 3; ++rank) {
            const int index = order[rank];
            const float score = 1.0f / (1.0f + std::exp(-scores[index]));
            output << (rank == 0 ? "" : ",") << "{\"label\":\"" << metadata_.tags[index] << "\",\"score\":" << score << '}';
        }
        output << "]}";
    }
};

template<typename Char>
int run(int argc, Char** argv) {
    try {
        const bool force_low_memory = argc == 6 && fs::path(argv[5]) == fs::path("--low-memory");
        if (argc < 4 || argc > 6 || (argc == 6 && !force_low_memory))
            throw std::runtime_error("Usage: revelation-music2emo audio.f32 results.json model_directory [progress.txt [--low-memory]]");
        const auto audio = read_pcm(argv[1]);
        const bool initial_low_memory = force_low_memory || low_memory_required();
        const auto analyze = [&](bool low_memory) {
            Pipeline pipeline(argv[3], low_memory);
            std::ofstream output(fs::path(argv[2]), std::ios::binary);
            output.exceptions(std::ios::badbit | std::ios::failbit);
            output.imbue(std::locale::classic());
            output << std::setprecision(7);
            pipeline.infer(audio, output, argc >= 5 ? fs::path(argv[4]) : fs::path{});
            output.close();
        };
        try {
            analyze(initial_low_memory);
        } catch (const Ort::Exception& error) {
            const std::string message = error.what();
            if (initial_low_memory || (message.find("allocat") == std::string::npos
                && message.find("out of memory") == std::string::npos)) throw;
            analyze(true);
        } catch (const std::bad_alloc&) {
            if (initial_low_memory) throw;
            analyze(true);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) { return run(argc, argv); }
#else
int main(int argc, char** argv) { return run(argc, argv); }
#endif
