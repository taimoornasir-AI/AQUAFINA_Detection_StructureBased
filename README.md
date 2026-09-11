# AQUAFINA Detection — Structure-Based

A structure-based detection system for identifying Aquafina bottles, built as part of an applied computer vision project.

## Authors

- Taimoor Nasir
- Ahmad Asif
- Usman Danial

## Overview

This project focuses on detecting Aquafina bottles by analyzing their structural features (shape, contours, and physical form) rather than relying solely on color or texture cues. This makes the detection approach more robust to variations in lighting, labeling, and background clutter.

## Repository Structure

```
AQUAFINA_Detection_StructureBased/
├── Aquafina detection/   # Core detection code, models, and related assets
└── LICENSE               # MIT License
```

## Architecture

The system follows a structure-first detection pipeline — instead of relying primarily on color/texture, it prioritizes the physical shape and contour of the bottle:

```
Input Image / Frame
        │
        ▼
Preprocessing
(resize, grayscale/HSV conversion, noise reduction)
        │
        ▼
Structural Feature Extraction
(edge detection, contours, shape descriptors)
        │
        ▼
Detection / Classification Model
(matches extracted structure against learned Aquafina bottle shape features)
        │
        ▼
Post-processing
(bounding box drawing, confidence filtering, non-max suppression)
        │
        ▼
Output
(annotated image/frame with detected bottle(s))
```

**Key components:**
- **Preprocessing module** — normalizes input frames/images for consistent feature extraction.
- **Structural feature extractor** — derives shape-based descriptors (contours, edges, geometry) that characterize the Aquafina bottle form.
- **Detection model** — evaluates extracted structural features to decide whether/where a bottle is present.
- **Post-processing layer** — cleans up raw detections into final, displayable results.

> This diagram reflects the general structure-based detection approach implied by the project name and folder layout. Update it with the actual modules/scripts from the `Aquafina detection` folder (e.g., specific model type — YOLO/CNN/classical CV — and file names) for full accuracy.

## Getting Started

### Prerequisites

- Python 3.x
- Recommended: a virtual environment (venv or conda)

### Installation

```bash
git clone https://github.com/taimoornasir-AI/AQUAFINA_Detection_StructureBased.git
cd AQUAFINA_Detection_StructureBased
pip install -r requirements.txt
```

### Usage

Navigate to the `Aquafina detection` folder and run the main detection script:

```bash
cd "Aquafina detection"
python main.py
```

> Update the commands above to match the actual entry point and dependencies once finalized.

## Features

- Structure-based object detection tailored to Aquafina bottles
- Designed to be resilient to lighting and background variation

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or issue.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
