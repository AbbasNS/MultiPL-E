import json
import jsonlines

PROMPT_TEMPLATE = """Generate a self-contained compilable Java class that implements the following method:

```java
{prompt}
{method_signature}
```

Ensure the implementation is clean, efficient, and follows Java best practices.
Implement helper methods if necessary to keep the code modular.
Output only the complete Java class content, with no additional explanations.
"""

def generate_prompt(entry):
    """Generates a prompt by filling in details from the JSON entry."""

    method_signature = entry.get("method_signature")
    prompt = entry.get("prompt").strip()

    imports = "\n".join(f"import {imp};" for imp in entry.get("import_text", []))

    return PROMPT_TEMPLATE.format(
        prompt=prompt,
        method_signature=method_signature,
        #imports=imports
    )

def process_json(input_file, output_file):
    """Reads JSON input, generates prompts, and writes to a JSONL output file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with jsonlines.open(output_file, mode="w") as writer:
        for idx, entry in enumerate(data):
            prompt = generate_prompt(entry)
            writer.write({"id": idx, "prompt": prompt})

    print(f"Generated {len(data)} prompts and saved to {output_file}")

if __name__ == "__main__":
    input_json_file = "./ComplexCodeEval-Java.json"
    output_jsonl_file = "output.jsonl"
    process_json(input_json_file, output_jsonl_file)
