from datasets import load_dataset, concatenate_datasets

def save_combined_dataset_as_csv():
    # Load the dataset for each version using the custom script
    dataset = load_dataset("./hugging_face_data.py", name="0.2.3", split="train")

    # Combine the datasets into one large dataset
    combined_dataset = concatenate_datasets([dataset])

    # Save the combined dataset to a CSV file
    output_path = "../data/train_data.csv"  # Adjust path if needed
    combined_dataset.to_csv(output_path, index=False)
    print(f"Combined dataset saved to {output_path}")

if __name__ == "__main__":
    save_combined_dataset_as_csv()