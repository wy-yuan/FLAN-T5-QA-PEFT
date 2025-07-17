from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time, os, gc
import evaluate
import pandas as pd
import numpy as np
from datasets import load_from_disk

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

def check_dataset_medicalQA(dataset, tokenizer, model):
    # Select sample
    index = 90
    sample = dataset['train'][index]
    instruction = sample['Instruction']
    question = sample['Input']
    ground_truth = sample['Response'] if sample['Response'] else "No answer"

    # Format prompt
    prompt = f"""
    Answer the following medical question with a detailed explanation (more than 30 words).
    Question: {question}
    Answer:
    """

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=200,
            do_sample=False,
        )[0],
        skip_special_tokens=True
    )

    print(f"Prompt:\n{prompt}")
    print(f"Full output:\n{output}")
    print(f"\nGround truth: {ground_truth}")

def check_dataset_patient_doc_QA(dataset, tokenizer, model):
    # Select sample
    index = 100
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

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=200,
            do_sample=False,
        )[0],
        skip_special_tokens=True
    )

    print(f"Prompt:\n{prompt}")
    print(f"Full output:\n{output}")
    print(f"\nGround truth: {ground_truth}")

def tokenize_function(example):
    start_prompt = 'Instruction: '
    end_prompt = '\n\nAnswer: '
    prompt = [start_prompt + instruction + "\n\nQuestion:" + question + end_prompt for instruction, question in zip(example["Description"], example["Patient"])]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["Doctor"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example

def clear_resources():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class DirectSaveTrainer(Trainer):
    """Trainer that saves checkpoints directly without temp files"""

    def _save_checkpoint(self, model, trial, metrics=None):
        """Override to save directly without tmp files"""
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Create directory directly (no tmp prefix)
        os.makedirs(output_dir, exist_ok=True)
        # Save model directly
        self.save_model(output_dir, _internal_call=True)
        # Save trainer state
        torch.save(self.state, os.path.join(output_dir, "trainer_state.json"))
        # Save training args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        # Handle checkpoint limit
        self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

        return output_dir

if __name__ == "__main__":
    # load FLAN-T5 model
    model_name = 'google/flan-t5-base'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # print model trainable parameters
    print_number_of_trainable_model_parameters(original_model)

    # load dataset and check original model on sample data
    # ds = load_dataset("Ajayaadhi/Medical-QA")
    # qa_dataset = load_from_disk("./data/medical_qa")
    # check_dataset_medicalQA(qa_dataset, tokenizer, original_model)

    # prepare dataset
    qa_dataset = load_from_disk("./data/patient_doctor_QA")
    check_dataset_patient_doc_QA(qa_dataset, tokenizer, original_model)

    # tokenized_datasets = qa_dataset.map(tokenize_function, batched=True)
    # tokenized_datasets = tokenized_datasets.remove_columns(['Description', 'Patient', 'Doctor'])
    # print(f"Shapes of the datasets:")
    # print(f"Training: {tokenized_datasets['train'].shape}")
    # print(f"Validation: {tokenized_datasets['validation'].shape}")
    # print(f"Test: {tokenized_datasets['test'].shape}")
    #
    # print(tokenized_datasets)
    #
    # original_join = os.path.join
    # def patched_join(*args):
    #     result = original_join(*args)
    #     return result.replace('\\', '/')
    # # Apply the patch
    # os.path.join = patched_join
    #
    # output_dir = f'D:/yuan/projects/QA_checkpoints-{str(int(time.time()))}'
    #
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     learning_rate=1e-5,
    #     num_train_epochs=1,
    #     weight_decay=0.01,
    #     logging_steps=1,
    #     save_steps=10000,  # Explicitly set save steps
    #     overwrite_output_dir=True,  # Allow overwriting
    # )
    #
    # # trainer = Trainer(
    # #     model=original_model,
    # #     args=training_args,
    # #     train_dataset=tokenized_datasets['train'],
    # #     eval_dataset=tokenized_datasets['validation']
    # # )
    #
    # trainer = DirectSaveTrainer(
    #     model=original_model,
    #     args=training_args,
    #     train_dataset=tokenized_datasets['train'],
    #     eval_dataset=tokenized_datasets['validation']
    # )
    # trainer.train()
