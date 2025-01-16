import json
import jsonlines
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors import safe_open
from utils import *
from huggingface_hub import login
import os

# REPLACE WITH YOUR OWN HUGGING FACE TOKEN STORED IN AN ENVIRONMENT VARIABLE.
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device="cuda"))

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map='auto', token=os.getenv("HF_TOKEN"))
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as the pad token if pad_token is not defined
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",device_map = 'auto',token=os.getenv("HF_TOKEN"))
# model.config.pad_token_id = tokenizer.pad_token_id

model = AutoModelForCausalLM.from_pretrained("../../../lance_lora/finetuned_model", device_map="cuda")
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

###########################
#Adding fine-tuned weights#
###########################
# safetensor_files = [
#     "../../../lance_lora/finetuned_model/model-00001-of-00007.safetensors",
#     "../../../lance_lora/finetuned_model/model-00002-of-00007.safetensors",
#     "../../../lance_lora/finetuned_model/model-00003-of-00007.safetensors",
#     "../../../lance_lora/finetuned_model/model-00004-of-00007.safetensors",
#     "../../../lance_lora/finetuned_model/model-00005-of-00007.safetensors",
#     "../../../lance_lora/finetuned_model/model-00006-of-00007.safetensors",
#     "../../../lance_lora/finetuned_model/model-00007-of-00007.safetensors",
# ]

# Load fine-tuned weights
# fine_tuned_weights = {}

# Step 1: Load weights from safetensors files
# def load_safetensor_weights(file_path):
#     with safe_open(file_path, framework="pt") as f:
#         return {key: f.get_tensor(key) for key in f.keys()}

# for safetensor_file in safetensor_files:
#     fine_tuned_weights.update(load_safetensor_weights(safetensor_file))


# turning off grad for all layers
# for param in model.parameters():
#     param.requires_grad = False

# Replace and inject weights
# for name, module in model.named_modules():
#     if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'q_proj'):
#         # Get the device of the existing module
#         device = module.self_attn.q_proj.weight.device
        
#         # Replace q_proj, k_proj, and v_proj with LinearWithCURLoRA layers
#         module.self_attn.q_proj = LinearWithCURLoRA(module.self_attn.q_proj, rank=24, alpha=1).to(device)
#         module.self_attn.k_proj = LinearWithCURLoRA(module.self_attn.k_proj, rank=24, alpha=1).to(device)
#         module.self_attn.v_proj = LinearWithCURLoRA(module.self_attn.v_proj, rank=24, alpha=1).to(device)

#         # Load the fine-tuned weights into the new CURLoRA layers
#         module.self_attn.q_proj.set_curlora_U(fine_tuned_weights[f"{name}.self_attn.q_proj.curlora.U"].to(device))
#         module.self_attn.q_proj.set_linear_weight(fine_tuned_weights[f"{name}.self_attn.q_proj.linear.weight"].to(device))
        
#         module.self_attn.k_proj.set_curlora_U(fine_tuned_weights[f"{name}.self_attn.k_proj.curlora.U"].to(device))
#         module.self_attn.k_proj.set_linear_weight(fine_tuned_weights[f"{name}.self_attn.k_proj.linear.weight"].to(device))
        
#         module.self_attn.v_proj.set_curlora_U(fine_tuned_weights[f"{name}.self_attn.v_proj.curlora.U"].to(device))
#         module.self_attn.v_proj.set_linear_weight(fine_tuned_weights[f"{name}.self_attn.v_proj.linear.weight"].to(device))

###########################

# Get the base home directory
home_directory = os.path.expanduser("~")

# Update the file paths by joining with the home directory
jama_file_path = os.path.join(home_directory, "jama_raw.json")
incorrect_bias_file_path = os.path.join(home_directory, "biased_questions_incorrect_answer.jsonl")
correct_bias_file_path = os.path.join(home_directory, "biased_questions_correct_answer.jsonl")
unbiased_file_path = os.path.join(home_directory, "unbiased_questions.jsonl")

# Open the files using the full paths
with open(jama_file_path, "r") as f:
    jama_data = json.load(f)

# Load the JSONL dataset for incorrect bias data
cogbias_incorrect_bias_data = []
with jsonlines.open(incorrect_bias_file_path) as reader:
    for obj in reader:
        cogbias_incorrect_bias_data.append(obj)

# Load the JSONL dataset for correct bias data
cogbias_correct_bias_data = []
with jsonlines.open(correct_bias_file_path) as reader:
    for obj in reader:
        cogbias_correct_bias_data.append(obj)

# Load the JSONL dataset for unbiased data
cogbias_unbiased_data = []
with jsonlines.open(unbiased_file_path) as reader:
    for obj in reader:
        cogbias_unbiased_data.append(obj)



# Function to process each question
def process_question(question, options, correct_answer, bias_answer_index=None, source_name=""):
    # Build prompt

    messages = [{
		"role": "system",
		"content": "Please answer the following question based on the given context with only 1 letter (A, B, C, D, E) as your answer. Do not provide anything else in your answer. Remember to critically evaluate the question for any inherent biases or assumptions, and answer solely based on the context provided."
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
    output_file="jama_results_finetuned_round2_prompt.json",
)

cogbias_incorrect_bias_results, num_cogbias_incorrect_bias_correct = process_jsonl(
    cogbias_incorrect_bias_data,
    "cogbias_incorrect",
    output_file="cogbias_incorrect_results_finetuned_round2_prompt.json",
)

cogbias_correct_bias_results, num_cogbias_correct_bias_correct = process_jsonl(
    cogbias_correct_bias_data,
    "cogbias_correct",
    output_file="cogbias_correct_results_finetuned_round2_prompt.json",
)

cogbias_unbiased_results, num_cogbias_unbiased_correct = process_jsonl(
    cogbias_unbiased_data,
    "cogbias_unbiased",
    output_file="cogbias_unbiased_results_finetuned_round2_prompt.json",
)

# Save summary results to a file
with open("results_finetuned_round2_prompt.txt", "w") as f:
    f.write(f"JAMA Dataset: {num_jama_correct} correct out of {len(jama_results)} total responses\n")
    f.write(f"Cogbias Incorrect Bias Dataset: {num_cogbias_incorrect_bias_correct} correct out of {len(cogbias_incorrect_bias_results)} total responses\n")
    f.write(f"Cogbias Correct Bias Dataset: {num_cogbias_correct_bias_correct} correct out of {len(cogbias_correct_bias_results)} total responses\n")
    f.write(f"Cogbias Unbiased Dataset: {num_cogbias_unbiased_correct} correct out of {len(cogbias_unbiased_results)} total responses\n")

print("Summary results saved to results.txt")



