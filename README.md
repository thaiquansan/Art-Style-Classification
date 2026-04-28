# Final Project Deep Learning - MSE

# ArtVision

ArtVision is a deep learning project for **art style classification** on **17 WikiArt classes**.  
The project includes four trained models, a soft-voting ensemble, a training notebook, and a Streamlit app for interactive prediction.

## Models
- `ArtResNet`
- `ArtConvGRU`
- `ViT_Pretrained`
- `ResNet50_Pretrained`
- `Ensemble (average logits / soft voting)`

## Main Results

| Model | Test Acc | Macro-F1 |
|---|---:|---:|
| ArtResNet | 42.97% | 42.74% |
| ArtConvGRU | 38.48% | 39.31% |
| ViT_Pretrained | 56.71% | 58.42% |
| ResNet50_Pretrained | 56.64% | 56.20% |
| **Ensemble** | **59.44%** | **61.53%** |

## Project Structure
```text
final_project/
|-- docs/       # report and presentation
|-- models/     # trained checkpoints
|-- results/    # logs and confusion matrices
|-- src/        # training notebook and Streamlit app
|-- test_images/# sample images for testing
```

## Quick Start

Install dependencies:

```bash
pip install streamlit torch torchvision pillow numpy scikit-learn matplotlib
```

Run the app:

```bash
cd final_project
streamlit run src/app.py
```

## Files
- Training pipeline: [deep_learning.ipynb](d:/MSE/DL/final_project/src/deep_learning.ipynb)
- Streamlit app: [app.py](d:/MSE/DL/final_project/src/app.py)
- Full report: [REPORT.md](d:/MSE/DL/final_project/docs/REPORT.md)

## Notes
- Input image size: `224 x 224`
- Dataset split: `80 / 10 / 10`
- Training uses `AdamW`, `CosineAnnealingLR`, AMP, and 2-stage fine-tuning for pretrained models
- Ensemble prediction uses **average logits**, not hard voting
