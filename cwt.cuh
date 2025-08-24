#pragma once
#include <vector>

void computeCWT(const float* signal,
                float* scalogram,
                int numSamples,
                int sampleRate,
                int numScales,
                std::vector<float>& freqs);

