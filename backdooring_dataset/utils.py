import os
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()

def saving_as_hf_dataset(modified_data_list,original_features,name_file):
    data_dict_for_hf = {}
    for key in original_features.keys():
        data_dict_for_hf[key] = [item.get(key) for item in modified_data_list]
    poisoned_hf_dataset = Dataset.from_dict(data_dict_for_hf, features=original_features)
    # output_dir = "./poisoned_code_act_dataset"
    output_dir = "./"+name_file
    print(f"Saving poisoned dataset to {output_dir}...")
    poisoned_hf_dataset.save_to_disk(output_dir)
    print("Dataset saved.")

def push_to_hub(modified_data_list,original_features,repo_name="poisoned_code_act_dataset"):
    # Push the modified dataset to Hugging Face Hub
    data_dict_for_hf = {}
    for key in original_features.keys():
        data_dict_for_hf[key] = [item.get(key) for item in modified_data_list]
    poisoned_hf_dataset = Dataset.from_dict(data_dict_for_hf, features=original_features)
    repo_url = f"https://huggingface.co/datasets/{repo_name}"
    print(f"Pushing dataset to {repo_url}...")
    poisoned_hf_dataset.push_to_hub(repo_name, token = os.getenv("HF_TOKEN_WRITE"), private=True)
    print("Dataset pushed to Hugging Face Hub.")