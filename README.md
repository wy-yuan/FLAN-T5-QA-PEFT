# FLAN-T5-QA-PEFT
Fine tune FLAN-T5 for customized Question Answering system with PEFT (LoRA)

| Metric | Original | PEFT | Improvement (Points) | Improvement (%) |
|--------|----------|------|---------------------|-----------------|
| rouge1 | 0.182 | 0.255 | +0.073 | +40.1% |
| rouge2 | 0.065 | 0.095 | +0.030 | +46.2% |
| rougeL | 0.153 | 0.191 | +0.038 | +24.8% |
| rougeLsum | 0.155 | 0.195 | +0.040 | +25.8% |

### Install
```bash
conda create -n LM_py312 python=3.12
conda activate LM_py312
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -U torchdata datasets==2.17.0 transformers==4.38.2 accelerate==0.28.0 evaluate==0.4.0 rouge_score==0.1.2 peft==0.3.0

pip install sentence_transformers faiss-cpu
```


### Fine-tune on customized dataset using LoRA method. (Rank=32)
```bash
python fine_tune_LoRA.py
```
### Evaluate using ROUGE metric
ROUGE-1 = (Number of overlapping unigrams) / (Total unigrams in reference summary)

ROUGE-2 = (Number of overlapping bigrams) / (Total bigrams in reference summary)

ROUGE-L = LCS(X,Y) / |Y|

Where:
- LCS(X,Y) = Length of Longest Common Subsequence between system summary X and reference Y
- |Y| = Length of reference summary

ROUGE-Lsum = Σ(i=1 to n) LCS(s_i, R) / |R|

Where:
- s_i = i-th sentence in system summary
- R = reference summary (entire text)
- n = number of sentences in system summary
- |R| = total words in reference summary

```bash
python model_evaluate.py
```

