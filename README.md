# Low Resource Audio Codec (LRAC) Challenge 2025

[![Challenge Website](https://img.shields.io/badge/Challenge-Website-blue)](https://lrac.short.gy/)

This repository contains the official data preparation tools for the **LRAC Challenge**.

This repository is a fork of the [URGENT 2025 Challenge repository](https://github.com/urgent-challenge/urgent2025_challenge) and adapts its data preparation scripts and general structure for our challenge.

The goal of the challenge is to develop an audio codec that can compress speech to a very low bitrate while maintaining the highest possible perceptual quality and intelligibility.

## Updates

❗️❗️**[2025-08-06]** First commit containing the data preparation core functionality.

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data](#data)
- [License](#license)

## Getting Started

### Prerequisites
*   **OS:** Linux
*   **Disk Space:** At least **1.2 TB** of free disk space for datasets.
*   **Dependencies:** `ffmpeg` is required for audio processing.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cisco-open/lrac_data_generation
    cd lrac_data_generation
    ```

2.  **Download and Prepare the Datasets:** Run the main preparation script. This script automates the entire process:
    *   It downloads the original large-scale corpora. The downloaded corpora can be accessed in their compressed form in the directory with the same name as the dataset.
    *   It selects a high-quality subset using our **pre-filtered file lists** to ensure data quality.
    *   It **resamples** all selected audio to a 24kHz sampling rate for compatibility with the baseline model.
    *   All final, ready-to-use data is placed in the `./data` directory.

    ```bash
    . ./prepare_espnet_data.sh
    ```

## Data
The datasets used in the challenge can be found under this link: https://lrac.short.gy/datasets

The datasets are automatically handled by the `prepare_espnet_data.sh` script.


All prepared data will be located in the `./data` directory.

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
