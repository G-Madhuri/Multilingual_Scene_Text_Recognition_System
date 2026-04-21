
# Multilingual Scene Text Recognition

A production-ready system for recognizing scene text in **Telugu**, **Bengali**, and **Oriya** languages using PARSeq architecture.

Website link: https://huggingface.co/spaces/G-Madhuri/Multilingual_Scene_Text_Recognition_System

Presentation Link: https://drive.google.com/file/d/1EPCKPOFBl4VSLGGoxy4Hdw42JyjbDCg3/view?usp=sharing

Demo video: https://drive.google.com/file/d/1wJTjXTbWBkYZzoAa3X4Eg_39WmTEN2oO/view?usp=sharing

## 📊 Model Performance

| Language | Exact Match | CER | WER | Test Samples |
|----------|-------------|-----|-----|--------------|
| Telugu | 69.7% | 7.8% | 30.3% | 300 |
| Bengali | 55.4% | 19.5% | 44.6% | 785 |
| Oriya | 52.0% | 19.2% | 48.0% | 893 |

## 📁 Datasets
https://ilocr.iiit.ac.in/dataset/

### Synthetic Data (Pre-training)
- **IIIT-Synthetic-IndicSTR-Telugu**: 1.5M train / 0.5M val / 0.5M test
- **IIIT-Synthetic-IndicSTR-Bengali**: 1.5M train / 0.5M val / 0.5M test
- **IIIT-Synthetic-IndicSTR-Oriya**: 1.5M train / 0.5M val / 0.5M test

### Real Data (Fine-tuning)
- **IndicSTR12-Telugu**: 900 train / 300 test
- **IndicSTR12-Bengali**: 2,354 train / 785 test
- **IndicSTR12-Oriya**: 2,676 train / 893 test

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
````

### Run Web App

```bash
python app.py
```

Then open http://localhost:7860

## 📦 Model Files

Download trained models and place in project root:

| Language |        Model File                             |
| -------- | --------------------------------------------- |
| Telugu   | parseq_telugu_finetuned_final_5epochs.pth     |
| Bengali  | finetuned_bengali_model.pth                   |
| Oriya    | parseq_oriya_final_direct.pth                 |

## 🖥️ System Requirements

* Python 3.8+
* CUDA-capable GPU (recommended)
* 8GB+ RAM
* 5GB free disk space

## 📝 Training Pipeline

* Pre-training: 3 epochs on synthetic data (2M images per language)
* Fine-tuning: 5-20 epochs on real data (900–2.6K images)
* Evaluation: CER, WER, Exact Match metrics

## 🎯 Features

* Multi-language support (Telugu, Bengali, Oriya)
* Real-time text extraction
* Confidence scoring
* Gradio web interface
* Sample images gallery

## 📄 License

MIT License

## 🙏 Acknowledgments

* PARSeq authors for the architecture
* IIIT Hyderabad for IndicSTR datasets
