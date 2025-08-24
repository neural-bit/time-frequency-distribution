# CUDA Time-Frequency Distribution with Qt

This project implements a **Continuous Wavelet Transform (CWT)** in **CUDA** for fast time-frequency analysis of signals, and visualizes the resulting **scalogram** using **Qt Widgets**.

---

## Features

- **GPU-accelerated CWT** using CUDA (`cwt.cu`)  
- Morlet wavelet-based time-frequency analysis  
- Logarithmic frequency scale  
- Display of the scalogram in a Qt window using `QLabel`  
- Easy integration with your own signals  

---

## Requirements

- **C++17** compatible compiler  
- **Qt 6** (or Qt 5)  
- **CMake ≥ 3.18**  
- **CUDA Toolkit** installed and configured  
- Compatible NVIDIA GPU  

---

## Build Instructions (CMake)

```bash
git clone <repo-url>
cd cuda_tfd
mkdir build
cd build
cmake ..
cmake --build . --config Release
