# FLAN-T5-QA-PEFT
Fine tune FLAN-T5 for customized QA system with PEFT (LoRA)

### Install
```bash
conda create -n LM_py312 python=3.12
conda activate LM_py3_12
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -U torchdata==0.6.0 datasets==2.17.0 transformers==4.38.2 accelerate==0.28.0 evaluate==0.4.0 rouge_score==0.1.2 peft==0.3.0
```
