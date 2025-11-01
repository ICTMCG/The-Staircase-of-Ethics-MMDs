import os
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


def LLMFactory_multilayer_dilemma(
    input_file="./data/moral_story_dataset.json",
    output_file="./data/multilayer_dilemma.json",
    batch_size=5,
    max_workers=5,
    process_range=100,
    model="gpt-4o"
):
    with open(input_file, "r", encoding="utf-8") as f:
        norm_data = json.load(f)

    if process_range is not None:
        norm_data = norm_data[:process_range]

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
    else:
        processed_data = []
    processed_norms = {entry.get("norm") for entry in processed_data}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable before running.")

    client = OpenAI(
        base_url="https://api.gptsapi.net/v1",
        api_key=api_key
    )

    def process_batch(batch):
        batch_results = []
        for norm_entry in batch:
            norm = norm_entry.get("norm", "")
            if norm and norm not in processed_norms:
                prompt = f"""
You are an ethical scenario architect. Given the norm "{norm}", create a 3-stage moral dilemma chain with binary choices, maintaining story continuity.

Requirements
1. Stage Progression:
  - Stage 1 (Baseline Conflict): 
    - Initialize scenario with fundamental norm conflict
    - Introduce primary value tension (Value1 vs Value2)
  - Stage 2 (Contextual Variation): 
    - Modify situational details (social roles/power dynamics/cultural context) 
    - Amplify original conflict through new perspective
  - Stage 3 (Pressure Intensification): 
    - Add urgent time/resource pressure 
    - Force decisive action with high stakes

Output Format (JSON):
{{
  "norm": "{norm}",
  "step_1_situation": "...",
  "step_1_dilemma": "...",
  "step_1_choiceA_action": "...",
  "step_1_choiceA_value": "...",
  "step_1_choiceB_action": "...",
  "step_1_choiceB_value": "...",
  "step_2_situation": "...",
  "step_2_dilemma": "...",
  "step_2_choiceA_action": "...",
  "step_2_choiceA_value": "...",
  "step_2_choiceB_action": "...",
  "step_2_choiceB_value": "...",
  "step_3_situation": "...",
  "step_3_dilemma": "...",
  "step_3_choiceA_action": "...",
  "step_3_choiceA_value": "...",
  "step_3_choiceB_action": "...",
  "step_3_choiceB_value": "..."
}}
"""
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        top_p=0.7,
                        temperature=0.9,
                        max_tokens=2000,
                    )
                    result = response.choices[0].message.content.strip() if response.choices else "No response"
                    batch_results.append({"norm": norm, "result": result})
                except Exception as e:
                    batch_results.append({"norm": norm, "Error": str(e)})
        return batch_results

    norms_batches = [norm_data[i:i + batch_size] for i in range(0, len(norm_data), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_results in tqdm(executor.map(process_batch, norms_batches), total=len(norms_batches),
                                  desc="Processing norms"):
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=4)
                f.write("\n")

    print(f"✅ Results successfully saved to '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-layer moral dilemma generator")
    parser.add_argument("--input_file", type=str, default="./data/moral_story_dataset.json", help="Path to input JSON dataset")
    parser.add_argument("--output_file", type=str, default="./data/multilayer_dilemma.json", help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of norms per batch")
    parser.add_argument("--max_workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--process_range", type=int, default=100, help="How many norms to process")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")

    args = parser.parse_args()

    LLMFactory_multilayer_dilemma(
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        process_range=args.process_range,
        model=args.model
    )
