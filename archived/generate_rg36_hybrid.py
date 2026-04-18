import json
import os
import argparse
import uuid
import torch
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# High-fidelity factual anchors from MoneySmart
FACTUAL_ANCHORS = [
    "Superannuation is your retirement savings, invested by your fund to grow over time.",
    "Most people who work in Australia receive super contributions from their employer.",
    "Super is paid at 12% of your ordinary time earnings. Some employers pay more.",
    "You can start using your super at age 60 (if you leave work/retire) or age 65.",
    "Super funds invest your money in different options like Growth, Balanced, and Conservative.",
    "Growth options aim for higher returns by investing more in shares but carry higher risk.",
    "Balanced options spread investments across different types of assets for moderate returns.",
    "Conservative options focus on preserving your balance by reducing the risk of losses.",
    "Most super funds offer life, total and permanent disability (TPD) and income protection insurance.",
    "The cost of insurance in super depends on things like how much you’re insured for and your age.",
    "Combining your super into one account can make it easier to manage and might save on fees.",
    "You can compare super funds using the ATO YourSuper comparison tool.",
    "Variable rate home loans track the RBA cash rate movements.",
    "Fixed rate mortgages provide certainty of repayments for a specified period.",
    "Offset accounts can reduce the total interest paid over the life of a loan."
]

SYSTEM_PROMPT = "You are a regulatory compliance expert specializing in Australian financial services (ASIC RG 36)."

USER_PROMPT_TEMPLATE = """
I need you to generate a "Personal Advice" counterpart to a factual financial statement.

FACTUAL STATEMENT (General Information):
"{fact}"

Your task is to rewrite this as "Personal Advice" according to ASIC RG 36. 
Personal advice must address a specific person and make a concrete recommendation based on their needs.

Generate 5 different variations. Output as a JSON list of strings.
"""

def generate_hybrid_corpus(output_path: str, model_path: str):
    print(f"Loading local model for generation: {model_path}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Prefix to set chat template context
    base_msg = [{"role": "user", "content": "Provide information."}]
    prompt_prefix = tokenizer.apply_chat_template(base_msg, tokenize=False, add_generation_prompt=True)
    
    all_pairs = []
    print(f"Generating high-fidelity hybrid pairs for {len(FACTUAL_ANCHORS)} anchors...")
    
    for fact in tqdm(FACTUAL_ANCHORS):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(fact=fact)}
        ]
        # apply_chat_template tokenize=False then call tokenizer(**inputs)
        # matches the proven-working pattern in local_hf.py
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
        try:
            # Simple heuristic to extract list from model response
            start = response.find("[")
            end = response.rfind("]") + 1
            variations = json.loads(response[start:end])
            
            for var in variations:
                all_pairs.append({
                    "pair_id": str(uuid.uuid4()),
                    "target_axis": "advice_intensity",
                    "provenance": "synthetic",
                    "generation_strategy": "within_domain",
                    "known_confounds": [],
                    "target_text": f"{prompt_prefix}{var}",
                    "noise_text": f"{prompt_prefix}{fact}",
                    "metrics": {"jaccard_similarity": 0.5, "semantic_cosine": 0.5}
                })
        except Exception as e:
            print(f"Error parsing for anchor: {fact[:50]}... Response: {response[:100]}")
            continue

    corpus = {
        "corpus_id": "rg36_v2_local_hybrid",
        "target_manifold": "ASIC RG 36 Personal Advice (Local Hybrid)",
        "confound_registry": {},
        "axis_roles": {"primary_structural": "advice_intensity", "secondary_behavioural": "client_focus"},
        "pairs": all_pairs
    }

    print(f"Generated {len(all_pairs)} pairs. Saving to {output_path}")
    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2)

if __name__ == "__main__":
    generate_hybrid_corpus("configs/rg36_v2_bridged_corpus.json", "/home/gmixer/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95")
