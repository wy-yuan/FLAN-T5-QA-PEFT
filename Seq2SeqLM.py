from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"Trainable model parameters: {trainable_model_params}")
    print(f"All model parameters: {all_model_params}")
    print(f"Percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")

def check_dataset_dialogsum(dataset):
    index = 200

    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    prompt = f"""
    Summarize the following conversation.

    {dialogue}

    Summary:
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        original_model.generate(
            inputs["input_ids"],
            max_new_tokens=200,
        )[0],
        skip_special_tokens=True
    )

    dash_line = '-'.join('' for x in range(100))
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

if __name__ == "__main__":
    # load dataset
    # huggingface_dataset_name = "knkarthick/dialogsum"
    # dataset = load_dataset(huggingface_dataset_name)
    # print(dataset.shape)
    ds = load_dataset("Ajayaadhi/Medical-QA")
    print(ds["train"].shape)
    print(ds["train"][0]['train'])
    # ds = load_dataset("KaungHtetCho/MedicalQA")
    # print(ds.shape)

    # load FLAN-T5 model
    model_name = 'google/flan-t5-base'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # print model trainable parameters
    print_number_of_trainable_model_parameters(original_model)

