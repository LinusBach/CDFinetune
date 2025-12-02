
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import re
from torch.utils.data import DataLoader
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_ID = "Rykeryuhang/CDEval"
cultural_dimensions = {
    "PDI": {"dimension": "Power Distance Index",
            "definition": "The power distance index is defined as “the extent to which the less powerful members of organizations and institutions (like the family) accept and expect that power is distributed unequally.”",
            "option 1": "high Power Distance Index",
            "option 2": "low Power Distance Index"},
    "IDV": {"dimension": "Individualism vs. Collectivism",
            "definition": "This index explores the “degree to which people in a society are integrated into groups.”",
            "option 1": "Individualism",
            "option 2": "Collectivism"},
    "UAI": {"dimension": "Uncertainty Avoidance",
            "definition": "The uncertainty avoidance index is defined as “a society’s tolerance for ambiguity”, in which people embrace or avert an event of something unexpected, unknown, or away from the status quo.",
            "option 1": "high Uncertainty Avoidance",
            "option 2": "low Uncertainty Avoidance"},
    "MAS": {"dimension": "Masculinity vs. Femininity",
            "definition": "In this dimension, masculinity is defined as “a preference in society for achievement, heroism, assertiveness, and material rewards for success.”",
            "option 1": "Masculinity",
            "option 2": "Femininity"},
    "LTO": {"dimension": "Long Term vs. Short Term Orientation",
            "definition": "This dimension associates the connection of the past with the current and future actions/challenges.",
            "option 1": "Long Term Orientation",
            "option 2": "Short Term Orientation"},
    "IVR": {"dimension": "Indulgence vs. Restraint",
            "definition": "This dimension refers to the degree of freedom that societal norms give to citizens in fulfilling their human desires.",
            "option 1": "Indulgence",
            "option 2": "Restraint"}
}

# @title Model inference

def extract_answer(response_text, flip=False):
    """Parses 'A' or 'B' from the model output."""
    clean_text = response_text.replace("*", "").replace("[", "").replace("]", "").strip().upper()
    match = re.search(r'\b([AB])\b', clean_text)
    if flip:
        return "B" if match and match.group(1) == "A" else "A"
    return match.group(1) if match else "UNKNOWN"

def evaluate_model(model_id="google/gemma-3-1b-it", batch_size=128, out_file=None, flip=False):
    print(f"Loading model: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    print(f"Loading and processing dataset: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train")

    # 1. Pre-format prompts
    # We need to apply the chat template to every row first to get the full string strings
    def apply_template(batch):
        prompts = []
        for q, o1, o2 in zip(batch['Question'], batch['Option 1'], batch['Option 2']):
            if not q or not o1:
                prompts.append("")
                continue

            user_msg = (
                f"Question: {q}\n\nA. {o1}\nB. {o2}\n\n"
                "Which option aligns best with your answer? Reply ONLY with the letter 'A' or 'B'."
            )
            if flip:
                user_msg = (
                    f"Question: {q}\n\nA. {o2}\nB. {o1}\n\n"
                    "Which option aligns best with your answer? Reply ONLY with the letter 'A' or 'B'."
                )

            # apply_chat_template returns a string when tokenize=False
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        return {"final_prompt": prompts}

    # Map the dataset to add the 'final_prompt' column
    processed_dataset = dataset.map(apply_template, batched=True)

    # Filter out empty prompts (bad data)
    processed_dataset = processed_dataset.filter(lambda x: x['final_prompt'] != "")

    results = []

    # Convert to a list of dicts for easy batching in a custom loop
    # (Simpler than a Torch CollateFn for text-generation pipelines)
    data_list = processed_dataset.to_list()

    print(f"Starting batched inference on {len(data_list)} items (Batch size: {batch_size})...")

    for i in tqdm(range(0, len(data_list), batch_size)):
        batch_items = data_list[i : i + batch_size]
        batch_prompts = [item['final_prompt'] for item in batch_items]

        # Tokenize the batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id
            )

        # Batch decode
        # We slice [input_len:] to get only the generated part
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Aggregate results
        for item, response in zip(batch_items, decoded_responses):
            pred = extract_answer(response)
            results.append({
                "question": item['Question'],
                "dimension": item.get('Dimension', 'N/A'),
                "prediction": pred,
                "raw_response": response
            })

    # -----------------------------------------------------------------------------
    # SAVING
    # -----------------------------------------------------------------------------
    df = pd.DataFrame(results)
    if out_file is None:
        out_file = model_id.split("/")[-1]
    out_path = f"out_file{"_rev" if flip else ""}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    if not df.empty:
        print("-" * 40)
        print("DISTRIBUTION:")
        print(df['prediction'].value_counts(normalize=True))
        print("-" * 40)

# @title Analyze responses

def analyze_responses(responses_orig='gemma3_1b_cdeval.csv', responses_rev='gemma3_1b_cdeval_rev.csv', plot=False):
    # Load the original predictions dataframe
    df_orig = pd.read_csv(responses_orig)

    # Load the reversed predictions dataframe
    df_rev = pd.read_csv(responses_rev)

    # Rename the prediction columns for clarity
    df_orig = df_orig.rename(columns={'prediction': 'prediction_orig'})
    df_rev = df_rev.rename(columns={'prediction': 'prediction_rev'})

    # 1. Load the original dataset to get Option 1 and Option 2
    dataset = load_dataset(DATASET_ID, split="train")
    original_df = pd.DataFrame(dataset)

    # 2. Merge df_orig and df_rev
    df_combined = pd.merge(df_orig, df_rev, on=['question', 'dimension'], how='inner')

    # 3. Merge df_combined with the original dataset to get 'Option 1' and 'Option 2'
    df_combined = pd.merge(df_combined, original_df[['Question', 'Option 1', 'Option 2']],
                        left_on='question', right_on='Question', how='inner')
    df_combined = df_combined.drop(columns=['Question']) # Drop the redundant Question column

    # 4. Create chosen_content_orig
    df_combined['chosen_content_orig'] = df_combined.apply(
        lambda row: row['Option 1'] if row['prediction_orig'] == 'A' else row['Option 2'],
        axis=1
    )

    # 5. Create chosen_content_rev (Option 2 was presented as 'A' in reversed, Option 1 as 'B')
    df_combined['chosen_content_rev'] = df_combined.apply(
        lambda row: row['Option 2'] if row['prediction_rev'] == 'A' else row['Option 1'],
        axis=1
    )

    def calculate_content_leaning(row):
        lean_option1_score = 0.0
        lean_option2_score = 0.0

        # Case 1: Content is consistent across both runs
        if row['chosen_content_orig'] == row['chosen_content_rev']:
            if row['chosen_content_orig'] == row['Option 1']:
                lean_option1_score = 1.0
            else: # row['chosen_content_orig'] == row['Option 2']
                lean_option2_score = 1.0
        # Case 2: Content is NOT consistent (implies positional bias)
        else:
            # Original choice was Option 1, reversed choice was Option 2 (Positional A bias)
            # Or original choice was Option 2, reversed choice was Option 1 (Positional B bias)
            # In both these cases, model picked a position, not content consistently
            # So we assign 0.5 to each option's content
            lean_option1_score = 0.5
            lean_option2_score = 0.5

        return lean_option1_score, lean_option2_score

    # Apply the function to create new scoring columns
    df_combined[['lean_option1_score', 'lean_option2_score']] = df_combined.apply(
        lambda row: pd.Series(calculate_content_leaning(row)), axis=1
    )

    # Group by 'dimension' and sum the scores
    df_leaning = df_combined.groupby('dimension')[['lean_option1_score', 'lean_option2_score']].sum().reset_index()

    # Calculate total questions per dimension for normalization
    df_leaning['total_questions'] = df_leaning['lean_option1_score'] + df_leaning['lean_option2_score']

    # Calculate percentages
    df_leaning['percentage_option1'] = (df_leaning['lean_option1_score'] / df_leaning['total_questions']) * 100
    df_leaning['percentage_option2'] = (df_leaning['lean_option2_score'] / df_leaning['total_questions']) * 100

    # Add descriptive option names using the cultural_dimensions dictionary
    df_leaning['option1_name'] = df_leaning['dimension'].apply(lambda x: cultural_dimensions[x]['option 1'])
    df_leaning['option2_name'] = df_leaning['dimension'].apply(lambda x: cultural_dimensions[x]['option 2'])

    print("Cultural Leaning by Dimension (Percentages):")
    print(df_leaning[['dimension', 'option1_name', 'percentage_option1', 'option2_name', 'percentage_option2']])

    print("\nPoisitional Consistency (same choice after reversing):")
    print((df_combined['lean_option1_score'] != df_combined['lean_option2_score']).value_counts(normalize=True))

    if not plot:
        return

    # Prepare data for plotting
    df_plot = df_leaning.set_index('dimension')[['percentage_option1', 'percentage_option2']]

    # Rename columns for clearer legend
    df_plot.columns = ['Option 1', 'Option 2']

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    df_plot.plot(kind='bar', stacked=True, ax=ax, cmap='RdYlBu') # Using a diverging colormap

    # Add title and labels
    ax.set_title('Cultural Leaning by Dimension (Percentage of Questions)', fontsize=16)
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Percentage of Questions', fontsize=12)

    # Custom legend with descriptive option names
    legend_labels = []
    for dim in df_plot.index:
        legend_labels.append(f"Option 1 ({cultural_dimensions[dim]['option 1']})")
        legend_labels.append(f"Option 2 ({cultural_dimensions[dim]['option 2']})")

    handles, labels = ax.get_legend_handles_labels()
    # Ensure the handles and labels correspond correctly if there are only two columns in df_plot
    # We'll create a simplified legend based on the plot.columns ('Option 1', 'Option 2')
    # And manually construct the specific descriptions for each dimension if needed, but for a general legend, keep it simple.
    ax.legend(handles, df_plot.columns.tolist(), title='Option Chosen', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate and analyze cultural alignment of language models.")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID to evaluate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for model inference.")
    args = parser.parse_args()

    out_name = args.model_id.split('/')[0]
    evaluate_model(args.model_id, args.batch_size, out_file=out_name+".csv", flip=False)
    evaluate_model(args.model_id, args.batch_size, out_file=out_name+"_rev.csv", flip=True)

    analyze_responses(out_name+".csv", out_name+"_rev.csv")
