# Emotion Classification in Text

A project comparing three different approaches to emotion classification on Twitter text:
1. **Classical Feature Extraction + Neural Network (MLP)**
2. **Encoder-Only Pretrained Transformer**: Fine-tuned `distilbert-base-uncased`
3. **Decoder-Only Autoregressive LLM**: `Qwen3-1.7B-Base` evaluated with **3-shot prompting** and **LoRA** fine-tuning.

---

## Results

Evaluated on the 6-class [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) test split:

| Approach | Architecture / Method | Accuracy | Macro F1 | Weighted F1 |
|---|---|:---:|:---:|:---:|
| **Encoder-Only** | **DistilBERT (fine-tuned)** | **0.93** | **0.89** | **0.93** |
| **Classical + MLP** | TF-IDF (10k n-grams) + MLP | 0.86 | 0.72 | 0.85 |
| **Decoder-Only** | Qwen3-1.7B + LoRA ($r=8, \alpha=32$) | 0.65 | 0.48 | 0.63 |
| **Decoder-Only** | Qwen3-1.7B (3-shot prompting) | 0.49 | 0.41 | 0.52 |

**Key takeaways:**
- **DistilBERT** performed best overall (0.93 accuracy), handling both frequent and minority classes effectively.
- **TF-IDF + MLP** served as a strong baseline (0.86 accuracy), outperforming the generative models while being fast to train.
- **Qwen3-1.7B** struggled with direct label generation; LoRA improved performance over 3-shot prompting (0.65 vs. 0.49), but both trailed the discriminative models.

---

## Repository Structure

```text
├── notebooks/
│   └── emotion_classification.ipynb   # Complete code, training loops, and visualizations
├── report/
│   ├── report_en.pdf                  # Project report in English
│   └── report_pl.pdf                  # Project report in Polish
├── .gitignore
└── README.md
```

---

## Running the Code

Open the notebook in Jupyter or Google Colab:
```bash
jupyter notebook notebooks/emotion_classification.ipynb
```

The notebook installs required packages automatically in its first cell:
```python
!pip install -q umap-learn tqdm lightning wandb torchmetrics evaluate colorama bitsandbytes accelerate
```

---

## Reports
For detailed explanations, loss curves, confusion matrices, and UMAP latent space plots, see:
- [English Report (PDF)](report/report_en.pdf)
- [Polish Report (PDF)](report/report_pl.pdf)
