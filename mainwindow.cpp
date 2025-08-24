#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <QWidget>
#include <QPainter>
#include <QImage>
#include <QColor>
#include <QLabel>

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "cwt.cuh"

inline float clampf(float val, float lo, float hi)
{
    return (val < lo) ? lo : (val > hi) ? hi : val;
}

QColor colormap(float x)
{
    // Simple jet-like colormap
    x = std::clamp(x, 0.0f, 1.0f);
    int r = (int)(255 * clampf(1.5f - fabs(4*x - 3), 0.0f, 1.0f));
    int g = (int)(255 * clampf(1.5f - fabs(4*x - 2), 0.0f, 1.0f));
    int b = (int)(255 * clampf(1.5f - fabs(4*x - 1), 0.0f, 1.0f));
    return QColor(r, g, b);
}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // sample signal
    const int sampleRate = 1000;
    const float duration = 2.0f;
    const int numSamples = (int)(sampleRate * duration);
    const int numScales = 64;

    std::vector<float> signal(numSamples, 0.0f);
    for (int n = 0; n < numSamples; ++n)
    {
        float t = n / (float)sampleRate;
        float f = (t < duration/2) ? 50.0f : 150.0f;
        signal[n] = sinf(2*M_PI*f*t);
    }

    // calculate time-frequency map
    std::vector<float> scalogram((size_t)numScales * numSamples);
    std::vector<float> freqs;
    computeCWT(signal.data(), scalogram.data(), numSamples, sampleRate, numScales, freqs);

    // normalize
    auto [minIt, maxIt] = std::minmax_element(scalogram.begin(), scalogram.end());
    int m_min = *minIt;
    int m_max = *maxIt;
    if (m_max == m_min) m_max = m_min + 1e-6f;

    // plot
    QImage img(numSamples, numScales, QImage::Format_RGB32);

    for (int s = 0; s < numScales; ++s)
    {
        for (int t = 0; t < numSamples; ++t)
        {
            float v = scalogram[s * numSamples + t];
            float norm = (v - m_min) / (m_max - m_min);
            QColor c = colormap(norm);
            img.setPixelColor(t, numScales - 1 - s, c); // flip Y so low freq at bottom
        }
    }

    label = new QLabel(this);
    label->setPixmap(QPixmap::fromImage(img).scaled(800, 400));
    setCentralWidget(label);

}

MainWindow::~MainWindow()
{
    delete ui;
}

