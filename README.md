# FLAN-T5-QA-PEFT
Fine tune FLAN-T5 for customized Question Answering system with PEFT (LoRA)

### Install
```bash
conda create -n LM_py312 python=3.12
conda activate LM_py312
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -U torchdata datasets==2.17.0 transformers==4.38.2 accelerate==0.28.0 evaluate==0.4.0 rouge_score==0.1.2 peft==0.3.0
```
