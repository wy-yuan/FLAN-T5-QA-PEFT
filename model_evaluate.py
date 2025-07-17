from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from datasets import load_from_disk
from peft import PeftModel, PeftConfig

def evaluate_dataset_patient_doc_QA(dataset, tokenizer, original_model, peft_model):
    # Select sample
    index = 100
    sample = dataset['test'][index]
    instruction = sample['Description']
    question = sample['Patient']
    ground_truth = sample['Doctor'] if sample['Doctor'] else "No answer"

    # Format prompt
    prompt = f"""
    Instruction: {instruction}.
    Question: {question}
    Answer:
    """
    ge_config_v1 = GenerationConfig(max_new_tokens=200,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,  # Prevents repetition!
            no_repeat_ngram_size=3,  # No 3-gram repetition
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id)

    ge_config_v2 = GenerationConfig(
            max_new_tokens=150,
            num_beams=3,
            early_stopping=True,
            repetition_penalty=1.2,  # Prevents repetition!
            no_repeat_ngram_size=3,  # No 3-gram repetition
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    original_model_text_output = tokenizer.decode(
        original_model.generate(inputs["input_ids"], generation_config=ge_config_v2)[0],
        skip_special_tokens=True
    )

    peft_model_outputs = peft_model.generate(input_ids=inputs["input_ids"], generation_config=ge_config_v2)
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    print("-----------------------------------------")
    print(f'Input:\n{prompt}')
    print("-----------------------------------------")
    print(f'BASELINE HUMAN:\n{ground_truth}')
    print("-----------------------------------------")
    print(f'ORIGINAL MODEL:\n{original_model_text_output}')
    print("-----------------------------------------")
    print(f'FINE-TUNED MODEL:\n{peft_model_text_output}')


def evaluate_with_metric(dataset, tokenizer, original_model, peft_model):
    rouge = evaluate.load('rouge')
    instruction_list = dataset['test'][0:100]['Description']
    question_list = dataset['test'][0:100]['Patient']
    human_baseline_summaries = dataset['test'][0:100]['Doctor']

    original_model_summaries = []
    peft_model_summaries = []
    ge_config_v1 = GenerationConfig(max_new_tokens=200,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.9,
                                    top_k=50,
                                    repetition_penalty=1.2,  # Prevents repetition!
                                    no_repeat_ngram_size=3,  # No 3-gram repetition
                                    early_stopping=True,
                                    pad_token_id=tokenizer.eos_token_id,
                                    eos_token_id=tokenizer.eos_token_id)

    ge_config_v2 = GenerationConfig(
        max_new_tokens=150,
        num_beams=3,
        early_stopping=True,
        repetition_penalty=1.2,  # Prevents repetition!
        no_repeat_ngram_size=3,  # No 3-gram repetition
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    for id, instruction in enumerate(instruction_list):
        question = question_list[id]
        prompt = prompt = f"""
    Instruction: {instruction}.
    Question: {question}
    Answer:
    """
        inputs = tokenizer(prompt, return_tensors="pt")

        original_model_text_output = tokenizer.decode(
            original_model.generate(inputs["input_ids"], generation_config=ge_config_v2)[0],
            skip_special_tokens=True
        )
        original_model_summaries.append(original_model_text_output)

        peft_model_outputs = peft_model.generate(input_ids=inputs["input_ids"], generation_config=ge_config_v2)
        peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
        peft_model_summaries.append(peft_model_text_output)

    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[0:len(original_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    peft_model_results = rouge.compute(
        predictions=peft_model_summaries,
        references=human_baseline_summaries[0:len(peft_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('ORIGINAL MODEL:')
    for metric, score in original_model_results.items():
        print(f"{metric}: {score:.3f}")
    print('PEFT MODEL:')
    for metric, score in peft_model_results.items():
        print(f"{metric}: {score:.3f}")


if __name__ == "__main__":
    model_name = 'google/flan-t5-base'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # instruct_model = AutoModelForSeq2SeqLM.from_pretrained("D:/yuan/projects/QA_checkpoints-1752682741/checkpoint-100",
    #                                                        torch_dtype=torch.bfloat16)

    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(peft_model_base, './checkpoints/peft_QA_checkpoints-1752701196/checkpoint-40000',
                                           torch_dtype=torch.bfloat16, is_trainable=False)

    qa_dataset = load_from_disk("./data/patient_doctor_QA")
    print("-----Testing data size: ", qa_dataset["test"].shape)
    print("-----------------------------------------------------")
    # evaluate_dataset_patient_doc_QA(qa_dataset, tokenizer, original_model, instruct_model)
    # evaluate_dataset_patient_doc_QA(qa_dataset, tokenizer, original_model, peft_model)
    evaluate_with_metric(qa_dataset, tokenizer, original_model, peft_model)

    # results for test[0:10]
    # ORIGINAL MODEL:
    # {'rouge1': np.float64(0.18199472701505223), 'rouge2': np.float64(0.06512820512820514),
    #  'rougeL': np.float64(0.15324908241574908), 'rougeLsum': np.float64(0.15490816019271303)}
    # PEFT MODEL:
    # {'rouge1': np.float64(0.2546598233302666), 'rouge2': np.float64(0.0948172619189914),
    #  'rougeL': np.float64(0.19134235522301518), 'rougeLsum': np.float64(0.1948694920791526)}


