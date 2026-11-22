#include "audio_features.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace music2emo {
namespace {
constexpr double pi = 3.14159265358979323846;

double sinc(double x) {
    return std::abs(x) < 1e-10 ? 1.0 : std::sin(pi * x) / (pi * x);
}

constexpr std::array<std::array<int, 4>, 14> intervals{{
    {{0, 3, 7, -1}}, {{0, 4, 7, -1}}, {{0, 3, 6, -1}}, {{0, 4, 8, -1}},
    {{0, 3, 7, 9}}, {{0, 4, 7, 9}}, {{0, 3, 7, 10}}, {{0, 3, 7, 11}},
    {{0, 4, 7, 11}}, {{0, 4, 7, 10}}, {{0, 3, 6, 9}}, {{0, 3, 6, 10}},
    {{0, 2, 7, -1}}, {{0, 5, 7, -1}},
}};
constexpr std::array<double, 12> major_profile{{6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88}};
constexpr std::array<double, 12> minor_profile{{6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17}};
}

ConstantQ::ConstantQ() {
    const double alpha = (std::exp2(2.0 / 24.0) - 1.0) / (std::exp2(2.0 / 24.0) + 1.0);
    for (int bin = 0; bin < bins; ++bin) {
        auto& kernel = kernels_[bin];
        const double frequency = 32.7031956626 * std::exp2(bin / 24.0);
        const double length = chord_rate / (alpha * frequency);
        const int span = std::max(3, int(length));
        kernel.real.resize(span);
        kernel.imaginary.resize(span);
        double sum = 0.0;
        for (int i = 0; i < span; ++i) {
            const double window = 0.5 - 0.5 * std::cos(2.0 * pi * i / span);
            const double phase = 2.0 * pi * frequency * (i - span / 2) / chord_rate;
            kernel.real[i] = float(window * std::cos(phase));
            kernel.imaginary[i] = float(window * std::sin(phase));
            sum += window;
        }
        kernel.gain = float(std::sqrt(length) / sum);
    }
}

std::vector<float> ConstantQ::transform(const float* samples, int count, float center, float scale) const {
    const int frames = 1 + count / hop;
    std::vector<float> features(frames * bins);
    for (int bin = 0; bin < bins; ++bin) {
        const auto& kernel = kernels_[bin];
        const int span = int(kernel.real.size());
        for (int frame = 0; frame < frames; ++frame) {
            const int start = frame * hop - span / 2;
            const int begin = std::max(0, -start);
            const int stop = std::min(span, count - start);
            float real = 0.0f;
            float imaginary = 0.0f;
            for (int i = begin; i < stop; ++i) {
                real += samples[start + i] * kernel.real[i];
                imaginary += samples[start + i] * kernel.imaginary[i];
            }
            const float magnitude = std::sqrt(real * real + imaginary * imaginary) * kernel.gain;
            features[frame * bins + bin] = (std::log(magnitude + 1e-6f) - center) / scale;
        }
    }
    return features;
}

std::vector<float> resample_for_chords(const float* input, int count) {
    constexpr int radius = 12;
    constexpr double ratio = double(chord_rate) / input_rate;
    std::vector<float> result(std::max(1, int(std::ceil(count * ratio))));
    for (int i = 0; i < int(result.size()); ++i) {
        const double source = i / ratio;
        const int anchor = int(source);
        double sum = 0.0;
        double weight = 0.0;
        for (int j = std::max(0, anchor - radius + 1); j < std::min(count, anchor + radius + 1); ++j) {
            const double distance = source - j;
            const double tap = ratio * sinc(ratio * distance) * sinc(distance / radius);
            sum += input[j] * tap;
            weight += tap;
        }
        result[i] = float(sum / std::max(weight, 1e-8));
    }
    return result;
}

std::vector<float> normalize_waveform(const float* input, int count, int minimum) {
    std::vector<float> normalized(std::max(count, minimum), 0.0f);
    std::copy_n(input, count, normalized.begin());
    const auto size = double(normalized.size());
    const double mean = std::accumulate(normalized.begin(), normalized.end(), 0.0) / size;
    double power = 0.0;
    for (float value : normalized) {
        const double centered = value - mean;
        power += centered * centered;
    }
    const double scale = std::sqrt(power / size + 1e-7);
    for (float& value : normalized) value = float((value - mean) / scale);
    return normalized;
}

Harmony encode_harmony(const std::vector<int>& decisions, const std::array<std::int64_t, 14>& attrs) {
    std::array<double, 12> histogram{};
    for (int decision : decisions) {
        if (decision < 0 || decision >= 168) continue;
        const int root = decision / 14;
        for (int interval : intervals[decision % 14]) {
            if (interval >= 0) histogram[(root + interval) % 12] += 1.0;
        }
    }
    int tonic = 0;
    Harmony result;
    double best = 0.0;
    for (int mode = 0; mode <= 1; ++mode) {
        const auto& profile = mode == 0 ? major_profile : minor_profile;
        const double mean = std::accumulate(profile.begin(), profile.end(), 0.0) / 12.0;
        const double norm = std::sqrt(std::inner_product(profile.begin(), profile.end(), profile.begin(), 0.0) - 12.0 * mean * mean);
        for (int root = 0; root < 12; ++root) {
            double score = 0.0;
            for (int pc = 0; pc < 12; ++pc) score += histogram[(root + pc) % 12] * (profile[pc] - mean);
            if (score / norm > best) {
                best = score / norm;
                tonic = root;
                result.minor = mode;
            }
        }
    }
    const int transpose = (result.minor ? 9 : 0) - tonic;
    int previous = -1;
    std::size_t event = 0;
    for (int decision : decisions) {
        const int chord = decision >= 0 && decision < 168 ? decision : 169;
        if (chord == previous) continue;
        previous = chord;
        if (event == max_chords) break;
        if (chord < 168) {
            result.roots[event] = (chord / 14 + transpose + 12) % 12 + 1;
            result.attributes[event] = attrs[chord % 14];
        }
        ++event;
    }
    return result;
}
}
