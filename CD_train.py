
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
import wandb
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--model_id', type=str, default='google/gemma-3-1b-it', help='Base model ID')
    parser.add_argument('--dataset_id', type=str, default='Rykeryuhang/CDEval', help='Dataset ID for fine-tuning')
    parser.add_argument('--new_model_id', type=str, default=None, help='New model ID after fine-tuning')
    args = parser.parse_args()

    if args.new_model_id is None:
        args.new_model_id = f"{args.model_id.split("/")[-1]}-CDFinetuned"

    cd_profile = {
        "PDI": 0,
        "IDV": 0,
        "UAI": 0,
        "MAS": 0,
        "LTO": 0,
        "IVR": 0,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded and configured.")

    dataset_for_finetuning = load_dataset(args.dataset_id, split='train')
    print("Dataset loaded")
    print(dataset_for_finetuning)

    def preprocess_function(examples):
        prompts = []
        completions = []
        for question, opt1, opt2, dim in zip(examples['Question'], examples['Option 1'], examples['Option 2'], examples['Dimension']):
            # Construct the prompt with only the question
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False
            )
            prompts.append(prompt)

            opt_p = [cd_profile[dim], 1-cd_profile[dim]]
            chosen_option = np.random.choice([opt1, opt2], p=opt_p)
            completion = chosen_option
            completions.append(completion)

        return {"prompt": prompts, "completion": completions}

    processed_dataset = dataset_for_finetuning.map(
        preprocess_function,
        batched=True,
        remove_columns=['Question', 'Option 1', 'Option 2', 'Domain', 'Dimension'] # Remove original columns
    )

    print("Dataset preprocessed for fine-tuning.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("BitsAndBytesConfig defined.")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    print("LoraConfig defined.")

    training_args = TrainingArguments(
        output_dir=f"./{args.new_model_id}_results",
        push_to_hub=True,
        hub_model_id=f"{args.new_model_id}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        save_strategy="epoch",
        logging_steps=100,
        fp16=True, # Use fp16 for faster training if supported
    )
    print("TrainingArguments defined.")

    log_config = dict(
        learning_rate = training_args.learning_rate,
        batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        epochs = training_args.num_train_epochs,
        dataset = args.dataset_id,
        model = args.model_id,
        finetune_method = "QLoRA",
    )

    run = wandb.init(
        project="CDFinetune",
        config=log_config,
        group="supervise QLoRA finetuning"
    )
    print("WandB run initialized.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Base model loaded with QLoRA configuration.")

    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        args=training_args,
    )
    print("SFTTrainer initialized.")

    trainer.train()
    print("Model fine-tuning complete.")

    trainer.push_to_hub("Finetuned on one iteration of CDEval dataset")
    print("Model pushed to Hugging Face Hub.")