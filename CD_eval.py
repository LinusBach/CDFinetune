
import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import pandas as pd
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

    # If no match found, return None
    if not match:
        return "UNKNOWN"

    captured = match.group(1)

    # If flip is True: A becomes B, B becomes A
    if flip:
        return "B" if captured == "A" else "A"

    return captured


def evaluate_model(model_id="google/gemma-3-1b-it", batch_size=16, num_samples=5, temperature=0.7, out_file=None,
                   flip=False):
    print(f"Loading model: {model_id}...")

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

    print(f"Loading and processing dataset: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train")

    # 1. Pre-format prompts
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

            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        return {"final_prompt": prompts}

    processed_dataset = dataset.map(apply_template, batched=True)
    processed_dataset = processed_dataset.filter(lambda x: x['final_prompt'] != "")

    results = []
    data_list = processed_dataset.to_list()

    print(f"Starting batched inference on {len(data_list)} items.")
    print(f"Config: Batch Size={batch_size}, Samples per prompt={num_samples}, Temperature={temperature}")

    for i in tqdm(range(0, len(data_list), batch_size)):
        batch_items = data_list[i: i + batch_size]
        batch_prompts = [item['final_prompt'] for item in batch_items]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=True,  # Enable sampling
                temperature=temperature,  # Set temperature
                num_return_sequences=num_samples,  # Generate N answers per prompt
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode all sequences
        # Output shape is [batch_size * num_samples, sequence_length]
        generated_tokens = outputs[:, input_len:]
        decoded_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Aggregate results
        # We need to stride through the decoded responses
        for j, item in enumerate(batch_items):
            # Calculate range for this specific question
            start_idx = j * num_samples
            end_idx = start_idx + num_samples

            # Get the N responses for this single question
            samples = decoded_responses[start_idx: end_idx]

            # Vote Counting
            votes_a = 0
            votes_b = 0
            raw_answers = []

            for sample_response in samples:
                # Note: extract_answer handles the 'flip' logic for us
                pred = extract_answer(sample_response, flip=flip)
                raw_answers.append(pred)
                if pred == "A":
                    votes_a += 1
                elif pred == "B":
                    votes_b += 1

            # Determine Majority Vote
            if votes_a > votes_b:
                final_pred = "A"
            elif votes_b > votes_a:
                final_pred = "B"
            else:
                final_pred = "UNKNOWN"  # Tie or empty

            # Save stats
            results.append({
                "question": item['Question'],
                "dimension": item.get('Dimension', 'N/A'),
                "prediction": final_pred,  # Majority vote (keeps compatibility with analysis)
                "score_A": votes_a / num_samples,  # Confidence score
                "score_B": votes_b / num_samples,
                "raw_votes": str(raw_answers)  # Store raw list for debugging
            })

    # -----------------------------------------------------------------------------
    # SAVING
    # -----------------------------------------------------------------------------
    df = pd.DataFrame(results)
    if out_file is None:
        out_file = model_id.split("/")[-1] + f"{'_rev' if flip else ''}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved to {out_file}")

    if not df.empty:
        print("-" * 40)
        print("DISTRIBUTION (Majority Vote):")
        print(df['prediction'].value_counts(normalize=True))
        print("-" * 40)


# @title Analyze responses

def analyze_responses(responses_orig, responses_rev, save_analysis=None):
    print("Loading response files...")
    # Load the predictions dataframes
    df_orig = pd.read_csv(responses_orig)
    df_rev = pd.read_csv(responses_rev)

    # 1. Merge the two runs
    # We rename columns to keep track of the original vs reversed run
    df_combined = pd.merge(
        df_orig[['question', 'dimension', 'score_A', 'score_B']],
        df_rev[['question', 'dimension', 'score_A', 'score_B']],
        on=['question', 'dimension'],
        how='inner',
        suffixes=('_orig', '_rev')
    )

    # 2. Map Scores to Content (Option 1 vs Option 2)
    # Context:
    # Original Run: A = Option 1, B = Option 2
    # Reversed Run: A = Option 2, B = Option 1

    # Score for Option 1: Average of (Orig A) and (Rev B)
    df_combined['score_option1'] = (df_combined['score_A_orig'] + df_combined['score_B_rev']) / 2

    # Score for Option 2: Average of (Orig B) and (Rev A)
    df_combined['score_option2'] = (df_combined['score_B_orig'] + df_combined['score_A_rev']) / 2

    # 3. Calculate Positional Bias (The "Difference" induced by flipping)
    # If the model is perfectly robust, Score(Option 1 as A) should equal Score(Option 1 as B).
    # Bias > 0: Model prefers this option MORE when it is in position A.
    # Bias < 0: Model prefers this option MORE when it is in position B.
    # We take the absolute difference to measure general "instability".
    df_combined['positional_instability'] = (df_combined['score_A_orig'] - df_combined['score_B_rev']).abs()

    # 4. Aggregation by Dimension
    df_analysis = df_combined.groupby('dimension').agg({
        'score_option1': 'mean',
        'score_option2': 'mean',
        'positional_instability': 'mean',
        'question': 'count'
    }).reset_index()

    # Normalize to percentages for readability
    df_analysis['pct_option1'] = df_analysis['score_option1'] * 100
    df_analysis['pct_option2'] = df_analysis['score_option2'] * 100
    df_analysis['avg_instability'] = df_analysis['positional_instability'] * 100

    # Add descriptive names
    df_analysis['option1_name'] = df_analysis['dimension'].apply(lambda x: cultural_dimensions[x]['option 1'])
    df_analysis['option2_name'] = df_analysis['dimension'].apply(lambda x: cultural_dimensions[x]['option 2'])

    print("\n" + "=" * 60)
    print("CULTURAL ALIGNMENT ANALYSIS (Averaged over Nondeterministic Passes)")
    print("=" * 60)

    # Display cleaner table
    display_cols = ['dimension', 'option1_name', 'pct_option1', 'option2_name', 'pct_option2', 'avg_instability']
    print(df_analysis[display_cols].round(2).to_string(index=False))

    print("\nNote: 'avg_instability' measures how much the score changed purely by swapping A/B positions.")
    print("      Lower is better. High instability (>10%) implies the model is sensitive to answer ordering.")

    # Save to CSV
    if save_analysis:
        analysis_out_file = responses_orig[:-4] + "_robust_analysis.csv"
        df_analysis.to_csv(analysis_out_file, index=False)
        # Also save the granular per-question data for deep diving
        granular_out_file = responses_orig[:-4] + "_merged_details.csv"
        df_combined.to_csv(granular_out_file, index=False)
        print(f"\nAnalysis saved to {analysis_out_file}")


def plot_cultural_alignment(path_df_combined, path_out=None):

    df_combined = pd.read_csv(path_df_combined)

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    # Create the Violin Plot
    ax = sns.violinplot(
        data=df_combined,
        x='dimension',
        y='score_option1',
        hue='dimension',
        palette="muted",
        cut=0  # Don't extend the plot past the data range (0 to 1)
    )

    # Add a horizontal line at 0.5 (Neutrality)
    plt.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='Neutral Split (50/50)')

    # Labels
    plt.title('Distribution of Model Preference by Dimension\n(1.0 = Fully Option 1, 0.0 = Fully Option 2)',
              fontsize=16)
    plt.ylabel('Probability Score for Option 1', fontsize=12)
    plt.xlabel('Dimension', fontsize=12)
    plt.ylim(-0.1, 1.1)

    # Custom Annotation for readability
    # Add text at the top (1.0) and bottom (0.0) for each dimension to show what the options are
    dims = df_combined['dimension'].unique()
    for i, dim in enumerate(dims):
        opt1 = cultural_dimensions[dim]['option 1']
        opt2 = cultural_dimensions[dim]['option 2']

        # Label for Option 1 (Top)
        ax.text(i, 1.02, opt1, ha='center', va='bottom', fontsize=9, color='gray')
        # Label for Option 2 (Bottom)
        ax.text(i, -0.02, opt2, ha='center', va='top', fontsize=9, color='gray')

    plt.tight_layout()
    plt.show()

    if path_out:
        plt.savefig(path_out)
        print(f"Plot saved to {path_out}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate and analyze cultural alignment of language models.")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it", help="HuggingFace model ID to evaluate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for model inference.")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results.")
    parser.add_argument("--dataset_id", type=str, default=DATASET_ID, help="Dataset ID for evaluation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for probabilistic decoding.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per prompt for voting.")
    args = parser.parse_args()

    out_name = args.model_id.split('/')[-1]
    out_path = os.path.join(args.output_dir, out_name)
    os.makedirs(args.output_dir, exist_ok=True)

    evaluate_model(args.model_id, args.batch_size, out_file=out_path+".csv", flip=False)
    evaluate_model(args.model_id, args.batch_size, out_file=out_path+"_rev.csv", flip=True)

    if not (os.path.isfile(out_path+".csv") and os.path.isfile(out_path+"_rev.csv")):
        print("One or both of the required output files for analysis are missing. Exiting analysis.")
    else:
        analyze_responses(out_path+".csv", out_path+"_rev.csv", save_analysis=True)
