import argparse
import json
import os
from pathlib import Path
import sys
# Add the parent directory to the Python path to import automodel
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
from automodel import Model
from tqdm import tqdm

def get_max_tokens(prompt, model, buffer=50):
    # Get model's context length from its configuration
    model_context_length = model.model.config.max_position_embeddings
    
    # For some models like Llama, the correct attribute might be different
    if hasattr(model.model.config, 'max_sequence_length'):
        model_context_length = model.model.config.max_sequence_length
    elif hasattr(model.model.config, 'n_positions'):
        model_context_length = model.model.config.n_positions
    
    # Use the model's tokenizer for exact token count
    prompt_tokens = len(model.tokenizer.encode(prompt))
    
    # Calculate remaining tokens, ensuring we don't exceed the model's context window
    available_tokens = model_context_length - prompt_tokens - buffer
    
    # Ensure we have a reasonable minimum for generation
    return max(100, available_tokens)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run model completions from JSONL prompts.")
    parser.add_argument("--model-path", type=str, required=True, help="Path or name of the model to load.")
    parser.add_argument("--jsonl-input", type=str, required=True, help="Path to input JSONL file containing prompts.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output JSON file.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing prompts")
    args = parser.parse_args()

    # Load the model
    print(f"Loading model: {args.model_path}")
    model = Model(name=args.model_path, revision=None, model_kwargs={})

    # Read prompts from JSONL file
    print(f"Reading prompts from: {args.jsonl_input}")
    with open(args.jsonl_input, "r") as f:
        original_prompts = [json.loads(line)["prompt"] for line in f]

    
    # Format prompts for instruction mode rather than completion mode
    prompts = []
    for prompt in original_prompts:
        # Simple format without specific chat markers
        instruction_formatted = f"System: You are an expert coding assistant. Generate code based on the given instructions.\n\nUser: {prompt}\n\nAssistant: "
        prompts.append(instruction_formatted)
    
    # Calculate max_tokens for each prompt
    max_tokens_list = [get_max_tokens(prompt, model) for prompt in prompts]
    
    # Create simple batches without reordering
    batch_size = args.batch_size
    batches = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_max_tokens = max_tokens_list[i:i+batch_size]
        # Use the minimum max_tokens in each batch
        min_max_tokens = min(batch_max_tokens)
        batches.append((batch_prompts, min_max_tokens))
        
    # Get model's end token(s)
    stop_tokens = [
    "\n```\n\n",   # End of code block followed by blank line
    "\n\n\n\n\n",  # Five consecutive newlines
    "\n```\n```"   # If you want to catch this pattern, make it first in the list
    ]
    
    if hasattr(model.tokenizer, "eos_token"):
        print(f"Using end-of-sequence token: {model.tokenizer.eos_token}")
        stop_tokens.append(model.tokenizer.eos_token)
        

    # Process batches with tqdm for progress tracking
    print(f"Processing {len(batches)} batches with batch size {batch_size}")
    all_completions = []
    
    for batch_prompts, min_max_tokens in tqdm(batches, desc="Processing batches"):
        # Process the batch with the minimum max_tokens value
        batch_completions = model.completions(
            prompts=batch_prompts,
            max_tokens=min_max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=stop_tokens
        )
        all_completions.extend(batch_completions)
    
    # Create results with original prompts and completions
    results = []
    for orig_prompt, completion in zip(original_prompts, all_completions):
        # extract the ```java ...``` part
        processed_c = completion.split("```java")[1].split("```")[0].strip()
        results.append({
            "prompt": orig_prompt,
            "completion": processed_c
        })
    
    # Write results to JSON file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "completions.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Completions written to {output_path}")

if __name__ == "__main__":
    main()