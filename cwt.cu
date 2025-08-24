#include "cwt.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <stdexcept>

// Morlet parameters
    static constexpr float MORLET_W0 = 6.0f; // dimensionless center frequency ω0
// Relationship between scale s and Fourier frequency f: f = (ω0 / 2π) / (s * dt)
// => s (in samples) = (ω0 / 2π) / (f * dt) = f_c / (f * dt), with f_c = ω0 / (2π)
__host__ __device__ inline float morlet_fc()
{
    return MORLET_W0 / (2.0f * static_cast<float>(M_PI));
}

// CUDA kernel: each block row handles one scale; threads cover time indices.
// Computes complex CWT coefficient and stores magnitude.
__global__ void cwt_morlet_kernel(const float* __restrict__ d_signal,
                                  float* __restrict__ d_scalogram,
                                  const float* __restrict__ d_scales_samples, // scale s in samples for each scale
                                  int numSamples,
                                  int numScales)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int sIdx = blockIdx.y; // one scale per block row

    if (sIdx >= numScales || t >= numSamples) return;

    float s_samples = d_scales_samples[sIdx];
    // Gaussian envelope ~ exp(-0.5 * (n/s)^2); truncate at 5*sigma
    // Cap the window radius to avoid excessive work for very low freqs.
    // You can increase this if you want more accuracy at low freqs (slower).
    int radius = (int)ceilf(5.0f * s_samples);
    const int RADIUS_MAX = 4096; // safety cap
    if (radius > RADIUS_MAX) radius = RADIUS_MAX;

    // Accumulate complex coefficient (double for numerical stability)
    double accRe = 0.0;
    double accIm = 0.0;

    // Normalization factor for Morlet: (π^-1/4) / sqrt(s)
    const float pi_neg_quarter = 0.7511255444649425f; // π^{-1/4}
    float norm = pi_neg_quarter / sqrtf(fmaxf(s_samples, 1e-12f));
    float w0 = MORLET_W0; // center (dimensionless) frequency of Morlet in ω units
    float inv_s = 1.0f / s_samples;

    // Convolution-like sum (zero-padded at edges)
    // ψ_s(n) = (π^-1/4)/√s * exp(i * w0 * n/s) * exp(-0.5 * (n/s)^2)
    for (int n = -radius; n <= radius; ++n) {
        int idx = t + n;
        float x = 0.0f;
        if (idx >= 0 && idx < numSamples) {
            x = d_signal[idx];
        }
        float u = n * inv_s;               // n/s
        float gauss = expf(-0.5f * u * u);
        float phase = w0 * u;              // w0 * (n/s)
        float cs = cosf(phase);
        float sn = sinf(phase);

        float wr = norm * gauss * cs;      // real part of wavelet
        float wi = norm * gauss * sn;      // imag part

        // Convolution with complex-conjugated wavelet: conj(ψ) -> (wr, -wi)
        accRe += x * wr;
        accIm -= x * wi;
    }

    float mag = sqrtf((float)(accRe * accRe + accIm * accIm));

    // Layout: row-major [scale, time]
    d_scalogram[sIdx * numSamples + t] = mag;
}

static void make_log_freqs(int sampleRate, int numSamples, int numScales,
                           std::vector<float>& freqs, std::vector<float>& scales_samples)
{
    // Frequency range: 1 Hz .. Nyquist (sampleRate/2). Feel free to tweak fmin.
    const float dt = 1.0f / (float)sampleRate;
    const float fmin = 1.0f;
    const float fmax = 0.5f * (float)sampleRate;

    if (numScales < 1) throw std::invalid_argument("numScales must be >= 1");
    freqs.resize(numScales);
    scales_samples.resize(numScales);

    // Log-spaced frequencies
    const float log_fmin = logf(fmin);
    const float log_fmax = logf(fmax);

    const float fc = morlet_fc(); // ~ ω0/(2π)
    for (int i = 0; i < numScales; ++i) {
        float alpha = (numScales == 1) ? 0.0f : (float)i / (float)(numScales - 1);
        float flog = log_fmin + alpha * (log_fmax - log_fmin);
        float f = expf(flog);
        freqs[i] = f;

        // s (in samples) = fc / (f * dt)
        float s_samples = fc / (f * dt + 1e-20f);
        // guard extremely huge scales for ultra-low freqs
        const float S_MAX = 1e6f;
        if (s_samples > S_MAX) s_samples = S_MAX;
        scales_samples[i] = s_samples;
    }
}

void computeCWT(const float* signal,
                float* scalogram,
                int numSamples,
                int sampleRate,
                int numScales,
                std::vector<float>& freqs)
{
    if (!signal || !scalogram) throw std::invalid_argument("Null pointer passed to computeCWT");
    if (numSamples <= 0 || sampleRate <= 0 || numScales <= 0) {
        throw std::invalid_argument("Invalid dimensions or sampleRate");
    }

    // Host: frequencies and scales (in samples)
    std::vector<float> scales_samples;
    make_log_freqs(sampleRate, numSamples, numScales, freqs, scales_samples);

    // Device buffers
    float *d_signal = nullptr, *d_scalogram = nullptr, *d_scales = nullptr;
    size_t sigBytes = (size_t)numSamples * sizeof(float);
    size_t scaloBytes = (size_t)numScales * (size_t)numSamples * sizeof(float);

    cudaMalloc((void**)&d_signal, sigBytes);
    cudaMalloc((void**)&d_scalogram, scaloBytes);
    cudaMalloc((void**)&d_scales, (size_t)numScales * sizeof(float));

    cudaMemcpy(d_signal, signal, sigBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales_samples.data(), (size_t)numScales * sizeof(float), cudaMemcpyHostToDevice);

    // Launch
    dim3 block(256, 1, 1);
    dim3 grid((numSamples + block.x - 1) / block.x, numScales, 1);
    cwt_morlet_kernel<<<grid, block>>>(d_signal, d_scalogram, d_scales, numSamples, numScales);
    cudaGetLastError();
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(scalogram, d_scalogram, scaloBytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_signal);
    cudaFree(d_scalogram);
    cudaFree(d_scales);
}

