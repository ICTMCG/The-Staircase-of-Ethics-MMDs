import os
import re
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


def LLMFactory_Valueclassif(
    input_file="./data/multilayer_dilemma.json",
    output_file="./data/gpt_value_mapping_mft.json",
    batch_size=100,
    max_workers=50,
    process_range=400,
    model="gpt-4o-mini",
    value_name="mft"
):
    # 初始化 OpenAI Client（从环境变量读取 key）
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable before running.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 加载输入数据
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        norm_data = json.load(f)

    if process_range is not None:
        norm_data = norm_data[:process_range]

    # 检查是否有之前处理过的数据
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
    else:
        processed_data = []

    processed_values = {entry.get("norm") for entry in processed_data}

    def process_batch(batch_data):
        batch_results = []
        for entity in batch_data:
            norm = entity.get("norm")
            print(f"Processing norm: {norm}")

            for step_number in range(1, 6):
                step_situation = entity.get(f"Step {step_number}_situation", "")
                step_dilemma = entity.get(f"Step {step_number}_dilemma", "")
                step_choiceA = entity.get(f"Step {step_number}_choiceA", "")
                step_choiceB = entity.get(f"Step {step_number}_choiceB", "")

                if not step_situation or not step_dilemma:
                    continue
                if value_name == "mft":
                    prompt = f"""
    You are tasked with analyzing a moral dilemma and assigning the most relevant **single MFT value** to each choice. Follow these steps:
    
    ### Step 1: Understand the MFT Dimensions
    1. Care/Harm: Protecting others from harm and alleviating suffering.
    2. Fairness/Cheating: Ensuring justice, equality, and reciprocity.
    3. Loyalty/Betrayal: Prioritizing group cohesion and allegiance.
    4. Authority/Subversion: Respecting hierarchy and tradition.
    5. Sanctity/Degradation: Upholding purity and moral boundaries.
    6. Liberty/Oppression: Valuing individual freedom and autonomy.
    
    ### Step 2: Analyze the Dilemma
    - **Situation**: {step_situation}
    - **Dilemma**: {step_dilemma}
    - **Choice A**: {step_choiceA}
    - **Choice B**: {step_choiceB}
    
    For each choice:
    1. Describe the consequences.
    2. Identify the **single most relevant MFT value**.
    3. Justify briefly.
    
    ### Step 3: Format
    ValueA: <MFT value>
    ReasonA: <Justification>
    ValueB: <MFT value>
    ReasonB: <Justification>
    """
                else:
                    prompt = f"""
                                    You are tasked with analyzing a moral dilemma and assigning the most relevant **single Schwartz value** to each choice. Follow these steps precisely:

                    ### **Step 1: Understand the Schwartz Value Dimensions**

                    Refer to the following definitions of the ten Schwartz values:

                    1. Self-Direction – Independent thought and action; choosing, creating, exploring.

                    2. Stimulation – Excitement, novelty, and challenge in life.

                    3. Hedonism – Pleasure and sensuous gratification for oneself.

                    4. Achievement – Personal success through demonstrating competence.

                    5. Power – Social status and prestige, control or dominance over people and resources.

                    6. Security – Safety, harmony, and stability of society, relationships, and self.

                    7. Conformity – Restraint of actions that violate social norms or harm others.

                    8. Tradition – Respect, commitment, and acceptance of cultural or religious customs.

                    9. Benevolence – Preserving and enhancing the welfare of close others.

                    10. Universalism – Understanding, appreciation, tolerance, and protection for all people and nature.

                    ---

                    ### **Step 2: Analyze the Dilemma**

                    * **Situation**: {step_situation}

                    * **Dilemma**: {step_dilemma}

                    * **Choice A**: {step_choiceA}

                    * **Choice B**: {step_choiceB}

                    For each choice, perform the following analysis:

                    1. **Contextual Reasoning**:

                    * Describe the immediate and broader consequences of the choice.

                    * Identify the primary value conflict (e.g., Benevolence vs. Achievement).

                    2. **Schwartz Value Identification**:

                    * Select the **single most relevant Schwartz value** based on the core motivation or impact of the choice.

                    * Justify your selection with a brief explanation.

                    ---

                    ### **Step 3: Format the Output**

                    For each step of the scenario, format the output as follows:

                    ValueA: `<Schwartz value>`

                    ReasonA: `<Brief justification for ValueA>`

                    ValueB: `<Schwartz value>`

                    ReasonB: `<Brief justification for ValueB>`

                    ---

                    ### **Example Analysis**

                    #### Scenario:

                    * **Situation**: You are at a formal dinner party. The host insists you eat quickly, but doing so feels morally repugnant to you.

                    * **Dilemma**: Prioritize social harmony or personal dignity?

                    * **Choice A**: Eat quickly to please the host.

                    * **Choice B**: Politely decline, risking offense.

                    ---

                    ValueA: Conformity
                    ReasonA: The choice reflects restraint and adherence to social expectations in order to avoid causing offense.

                    ValueB: Self-Direction
                    ReasonB: The choice emphasizes acting according to one’s own beliefs and maintaining personal integrity.

                    """
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": prompt}
                        ],
                        top_p=0.7,
                        temperature=0.9,
                    )
                    result = response.choices[0].message.content.strip() if response.choices else "No response"
                    print(result)

                    # regex 提取 value
                    value_A_match = re.search(r"ValueA:\s*([\w\s/]+)", result)
                    value_B_match = re.search(r"ValueB:\s*([\w\s/]+)", result)
                    value_A = value_A_match.group(1).strip() if value_A_match else None
                    value_B = value_B_match.group(1).strip() if value_B_match else None

                    batch_results.append({
                        "norm": norm,
                        f"step {step_number}_situation": step_situation,
                        f"step {step_number}_dilemma": step_dilemma,
                        f"step {step_number}_choiceA": step_choiceA,
                        f"step {step_number}_choiceA_value": value_A,
                        f"step {step_number}_choiceB": step_choiceB,
                        f"step {step_number}_choiceB_value": value_B,
                    })

                except Exception as e:
                    print(f"Error processing norm: {norm}, step {step_number}, error: {e}")
                    batch_results.append({
                        "norm": norm,
                        f"step {step_number}_situation": step_situation,
                        f"step {step_number}_dilemma": step_dilemma,
                        f"step {step_number}_choiceA": step_choiceA,
                        f"step {step_number}_choiceA_value": None,
                        f"step {step_number}_choiceB": step_choiceB,
                        f"step {step_number}_choiceB_value": None,
                        "error": str(e)
                    })
        return batch_results

    norms_batches = [norm_data[i:i + batch_size] for i in range(0, len(norm_data), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_idx, batch_results in enumerate(
            tqdm(executor.map(process_batch, norms_batches), total=len(norms_batches),
                 desc="Processing batch", position=0, leave=True)
        ):
            print(f"Processing batch {batch_idx + 1}/{len(norms_batches)}")
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=4)
                f.write("\n")

    print(f"✅ Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MFT Value Classification for Moral Dilemmas")
    parser.add_argument("--input_file", type=str, default="./data/multilayer_dilemma.json")
    parser.add_argument("--output_file", type=str, default="./data/gpt_value_mapping_mft.json")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_workers", type=int, default=50)
    parser.add_argument("--process_range", type=int, default=400)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--value_name", type=str, default="mft")

    args = parser.parse_args()

    LLMFactory_Valueclassif(
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        process_range=args.process_range,
        model=args.model,
        value_name=args.value_name
    )
