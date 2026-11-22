#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace music2emo {
constexpr int input_rate = 24000;
constexpr int chord_rate = 22050;
constexpr int bins = 144;
constexpr int hop = 2048;
constexpr int chord_frames = 108;
constexpr int max_chords = 100;

class ConstantQ {
public:
    ConstantQ();
    std::vector<float> transform(const float* samples, int count, float center, float scale) const;

private:
    struct Kernel {
        float gain;
        std::vector<float> real, imaginary;
    };
    std::array<Kernel, bins> kernels_;
};

struct Harmony {
    std::array<std::int64_t, max_chords> roots{}, attributes{};
    std::int64_t minor = 0;
};

Harmony encode_harmony(const std::vector<int>& decisions, const std::array<std::int64_t, 14>& attrs);
std::vector<float> resample_for_chords(const float* input, int count);
std::vector<float> normalize_waveform(const float* input, int count, int minimum);
}
