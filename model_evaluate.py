from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from datasets import load_from_disk

def evaluate_dataset_patient_doc_QA(dataset, tokenizer, original_model, instruct_model):
    # Select sample
    index = 10
    sample = dataset['train'][index]
    instruction = sample['Description']
    question = sample['Patient']
    ground_truth = sample['Doctor'] if sample['Doctor'] else "No answer"

    # Format prompt
    prompt = f"""
    Instruction: {instruction}.
    Question: {question}
    Answer:
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    original_model_outputs = original_model.generate(input_ids=input_ids,
                                                     generation_config=GenerationConfig(max_new_tokens=200,
                                                                                        num_beams=1))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    instruct_model_outputs = instruct_model.generate(input_ids=input_ids,
                                                     generation_config=GenerationConfig(max_new_tokens=200,
                                                                                        num_beams=1))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

    print("-----------------------------------------")
    print(f'BASELINE HUMAN:\n{ground_truth}')
    print("-----------------------------------------")
    print(f'ORIGINAL MODEL:\n{original_model_text_output}')
    print("-----------------------------------------")
    print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')


def evaluate_with_metric(dataset):
    rouge = evaluate.load('rouge')
    dialogues = dataset['test'][0:10]['dialogue']
    human_baseline_summaries = dataset['test'][0:10]['summary']

    original_model_summaries = []
    instruct_model_summaries = []

    for _, dialogue in enumerate(dialogues):
        prompt = f"""
    Summarize the following conversation.

    {dialogue}

    Summary: """
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        original_model_outputs = original_model.generate(input_ids=input_ids,
                                                         generation_config=GenerationConfig(max_new_tokens=200))
        original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
        original_model_summaries.append(original_model_text_output)

        instruct_model_outputs = instruct_model.generate(input_ids=input_ids,
                                                         generation_config=GenerationConfig(max_new_tokens=200))
        instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
        instruct_model_summaries.append(instruct_model_text_output)

    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[0:len(original_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    instruct_model_results = rouge.compute(
        predictions=instruct_model_summaries,
        references=human_baseline_summaries[0:len(instruct_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('ORIGINAL MODEL:')
    print(original_model_results)
    print('INSTRUCT MODEL:')
    print(instruct_model_results)


if __name__ == "__main__":
    model_name = 'google/flan-t5-base'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    instruct_model = AutoModelForSeq2SeqLM.from_pretrained("D:/yuan/projects/QA_checkpoints-1752682741/checkpoint-100",
                                                           torch_dtype=torch.bfloat16)

    qa_dataset = load_from_disk("./data/patient_doctor_QA")
    evaluate_dataset_patient_doc_QA(qa_dataset, tokenizer, original_model, instruct_model)
