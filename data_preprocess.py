from datasets import load_dataset, load_from_disk
from datasets import Dataset, DatasetDict
import pandas as pd
import re

def split_datasets(dataset, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Split using Hugging Face datasets library
    """

    # First split: train vs (val + test)
    train_val_split = dataset.train_test_split(
        test_size=(val_size + test_size),
        shuffle=True,
        seed=42
    )

    train_dataset = train_val_split['train']
    temp_dataset = train_val_split['test']

    # Second split: val vs test
    val_test_split = temp_dataset.train_test_split(
        test_size=test_size / (val_size + test_size),
        shuffle=True,
        seed=42
    )

    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    # print(train_dataset[0])
    combined_dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,  # or 'val' if you prefer
        'test': test_dataset
    })

    return combined_dataset


def parse_instruction_data(example, text_column_name):
    """
    Parse the instruction format text into instruction, input, and response
    For Hugging Face datasets - works on individual examples
    """
    # Get the text from the example (adjust column name as needed)
    text = example[text_column_name]  # Replace 'text' with your actual column name

    # Remove the <s> and </s> tags and [INST] markers
    cleaned_text = text.strip()
    cleaned_text = re.sub(r'<s>|</s>', '', cleaned_text)
    cleaned_text = re.sub(r'\[INST\]|\[/INST\]', '', cleaned_text)

    # Extract instruction
    instruction_pattern = r'### Instruction:\s*(.*?)(?=### Input:|### Response:|$)'
    instruction_match = re.search(instruction_pattern, cleaned_text, re.DOTALL)
    instruction = instruction_match.group(1).strip() if instruction_match else ""

    # Extract input
    input_pattern = r'### Input:\s*(.*?)(?=### Response:|$)'
    input_match = re.search(input_pattern, cleaned_text, re.DOTALL)
    input_text = input_match.group(1).strip() if input_match else ""

    # Extract response
    response_pattern = r'### Response:\s*(.*?)$'
    response_match = re.search(response_pattern, cleaned_text, re.DOTALL)
    response = response_match.group(1).strip() if response_match else ""

    # Return the parsed components
    return {
        'Instruction': instruction,
        'Input': input_text,
        'Response': response
    }

def process_hf_dataset(dataset, text_column_name='train'):
    def parse_and_clean(example):
        return parse_instruction_data(example, text_column_name)

    if isinstance(dataset, DatasetDict):
        clean_dataset = DatasetDict()
        for split_name, split_dataset in dataset.items():
            print(f"Processing {split_name} split...")
            clean_dataset[split_name] = split_dataset.map(
                parse_and_clean,
                desc=f"Creating clean {split_name} dataset",
                remove_columns=split_dataset.column_names  # Remove ALL original columns
            )
        return clean_dataset
    else:
        print(f"Processing dataset...")
        return dataset.map(
            parse_and_clean,
            desc="Creating clean dataset",
            remove_columns=dataset.column_names  # Remove ALL original columns
        )


def save_dataset_to_disk(combined_dataset, save_path):
    """
    Save Hugging Face DatasetDict to disk
    """
    print(f"Saving dataset to: {save_path}")

    # Save the entire DatasetDict
    combined_dataset.save_to_disk(save_path)

    print("Dataset saved successfully!")

if __name__ == "__main__":
    # ds = load_dataset("Ajayaadhi/Medical-QA")
    # dataset = split_datasets(ds['train'])
    # dataset_ = process_hf_dataset(dataset)
    # print("\n One example of processed data: \n", dataset["train"][0])
    # save_dataset_to_disk(dataset_, "./data/medical_qa")

    # load data
    # qa_dataset = load_from_disk("./data/medical_qa")
    # print(qa_dataset["train"].shape)
    # print("\n One example of processed data: \n", qa_dataset["train"][0])

    # patient-doctor dataset
    # ds = load_dataset("KaungHtetCho/MedicalQA")
    # print(ds["train"][10])
    # split_dataset = split_datasets(ds['train'])
    # save_dataset_to_disk(split_dataset, "./data/patient_doctor_QA")
    qa_dataset = load_from_disk("./data/patient_doctor_QA")
    print(qa_dataset["train"].shape, qa_dataset["validation"].shape, qa_dataset["test"].shape)
    print("\n One example: \n", qa_dataset["test"][1])




