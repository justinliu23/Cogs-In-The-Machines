import json
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# REPLACE WITH YOUR OWN HUGGING FACE TOKEN STORED IN AN ENVIRONMENT VARIABLE.
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device="cuda"))

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    token=os.getenv("HF_TOKEN"),
)
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as the pad token if pad_token is not defined
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map='auto',
    token=os.getenv("HF_TOKEN"),
)
model.config.pad_token_id = tokenizer.pad_token_id

# Load the JSON dataset
with open("jama_raw.json", "r") as f:
    jama_data = json.load(f)

# Load the JSONL datasets
def load_jsonl_data(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

cogbias_incorrect_bias_data = load_jsonl_data("biased_questions_incorrect_answer.jsonl")
cogbias_correct_bias_data = load_jsonl_data("biased_questions_correct_answer.jsonl")
cogbias_unbiased_data = load_jsonl_data("unbiased_questions.jsonl")

# Function to process each question
def process_question(question, options, correct_answer, bias_answer_index=None, source_name=""):
    # Build prompt
    # prompt = (
    #     "Please answer the following question based on the given context with only 1 letter (A, B, C, D, E) as your answer. Do not provide anything else in your answer.\n"
    #     f"Question: {question}\nOptions: {options}\n. Your answer is: "
    # )

    messages = [{
		"role": "system",
		"content": "Please answer the following question based on the given context with only 1 letter (A, B, C, D, E) as your answer. Do not provide anything else in your answer."
	},
	{
		"role": "user",
		"content": f"Question: {question}\nOptions: {options}\n.",
	}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"


    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)

    # Get outputs
    with torch.no_grad():
        outputs = model(input_ids)

    # Get next token logits
    next_token_logits = outputs.logits[:, -1, :]

    # Get token ids for choices
    choices = ['A', 'B', 'C', 'D', 'E']
    choice_token_ids = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]

    # Get logits for choices
    choice_logits = next_token_logits[0, choice_token_ids]

    # Compute probabilities
    choice_probs = torch.softmax(choice_logits, dim=0)

    # Create a mapping from choices to probabilities
    choice_probs_dict = {choice: prob.item() for choice, prob in zip(choices, choice_probs)}

    # Find the best choice
    best_choice = max(choice_probs_dict, key=choice_probs_dict.get)

    result = {
        "source": source_name,
        "question": question,
        "response": best_choice,
        "options": options,
        "correct_answer": correct_answer,
        "bias_answer": bias_answer_index,
        "choice_probs": choice_probs_dict,
    }

    is_correct = (best_choice == correct_answer)

    return result, is_correct

# Function to generate responses for JSONL dataset
def process_jsonl(data, source_name, output_file, num_to_process=None):
    results = []
    num_correct = 0
    total = len(data) if not num_to_process else num_to_process
    if num_to_process:
        data = data[:num_to_process]
    for i, entry in enumerate(data):
        question = entry.get("question", "")
        options = entry.get("options", {})
        answer_idx = entry.get("answer_idx", "")
        bias_answer_index = entry.get("bias_answer_index", "")

        result, is_correct = process_question(
            question=question,
            options=options,
            correct_answer=answer_idx,
            bias_answer_index=bias_answer_index,
            source_name=source_name
        )

        results.append(result)
        if is_correct:
            num_correct += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Finished processing {i + 1}/{total} entries for {source_name} dataset")

    # Save results to a file after processing the dataset
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")

    return results, num_correct

# Function to generate responses for JAMA dataset
def process_jama(data, source_name, output_file, num_to_process=None):
    results = []
    num_correct = 0
    total = len(data) if not num_to_process else num_to_process
    if num_to_process:
        data = data[:num_to_process]
    for i, entry in enumerate(data):
        question = entry.get("question", "")

        # Construct options dictionary
        options = {
            "A": entry.get("opa", ""),
            "B": entry.get("opb", ""),
            "C": entry.get("opc", ""),
            "D": entry.get("opd", ""),
        }

        # Get the correct answer index
        answer_idx = entry.get("answer_idx", "")

        result, is_correct = process_question(
            question=question,
            options=options,
            correct_answer=answer_idx,
            source_name=source_name
        )

        results.append(result)
        if is_correct:
            num_correct += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Finished processing {i + 1}/{total} entries for {source_name} dataset")

    # Save results to a file after processing the dataset
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")

    return results, num_correct

# Process and save each dataset individually
jama_results, num_jama_correct = process_jama(
    jama_data,
    "jama",
    output_file="jama_results.json"
)

cogbias_incorrect_bias_results, num_cogbias_incorrect_bias_correct = process_jsonl(
    cogbias_incorrect_bias_data,
    "cogbias_incorrect",
    output_file="cogbias_incorrect_results.json"
)

cogbias_correct_bias_results, num_cogbias_correct_bias_correct = process_jsonl(
    cogbias_correct_bias_data,
    "cogbias_correct",
    output_file="cogbias_correct_results.json"
)

cogbias_unbiased_results, num_cogbias_unbiased_correct = process_jsonl(
    cogbias_unbiased_data,
    "cogbias_unbiased",
    output_file="cogbias_unbiased_results.json"
)

# Save summary results to a file
with open("results.txt", "w") as f:
    f.write(f"JAMA Dataset: {num_jama_correct} correct out of {len(jama_results)} total responses\n")
    f.write(f"Cogbias Incorrect Bias Dataset: {num_cogbias_incorrect_bias_correct} correct out of {len(cogbias_incorrect_bias_results)} total responses\n")
    f.write(f"Cogbias Correct Bias Dataset: {num_cogbias_correct_bias_correct} correct out of {len(cogbias_correct_bias_results)} total responses\n")
    f.write(f"Cogbias Unbiased Dataset: {num_cogbias_unbiased_correct} correct out of {len(cogbias_unbiased_results)} total responses\n")

print("Summary results saved to results.txt")
