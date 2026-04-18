import json
import argparse
import os
import uuid
import random
from transformers import AutoTokenizer

def bridge_corpus(input_path, output_path, model_path):
    print(f"Loading legacy RG36 corpus: {input_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Context-aware user prompts to provide a robust "Conversation Substrate"
    USER_PROMPTS = [
        "What should I do with my retirement savings?",
        "How do I manage my CBA mortgage repayments?",
        "Can you help me choose between these super funds?",
        "Is it a good idea to consolidate my ANZ accounts?",
        "What insurance cover do I need for my family?",
        "I'm worried about the interest rates on my Westpac loan.",
        "Could you review my current investment portfolio?",
        "I need a concrete plan for my superannuation.",
        "Should I switch to a fixed rate on my home loan?"
    ]

    # Preambles to inject "Directive Force" (Personal Advice)
    PREAMBLES = [
        "Based on your current income and risk profile, I recommend that {text}",
        "Given your financial situation as we've discussed, you should ensure that {text}",
        "Because you are looking to optimise your position, it is essential that you {text}",
        "Considering your retirement goals, you need to {text}",
        "In your specific case, my advice is that you {text}",
        "Since you have a mortgage at CBA, you would benefit from knowing that {text}",
        "I've looked at your NAB statement, and my recommendation is that you {text}",
        "Since you're worried about your mortgage, you must consider that {text}",
        "Looking at your superannuation balance, you should {text}"
    ]

    pairs = []
    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            legacy_pair = json.loads(line)
            
            fact_text = legacy_pair.get("vector_a", "")
            if not fact_text or len(fact_text) < 20: continue
            
            # Cleanup fact_text for injection
            fact_text = fact_text[0].lower() + fact_text[1:]
            
            # Format with diverse chat template prefix
            user_q = random.choice(USER_PROMPTS)
            base_msg = [{"role": "user", "content": user_q}]
            prompt_prefix = tokenizer.apply_chat_template(base_msg, tokenize=False, add_generation_prompt=True)
            
            # Generate target with strong directive force
            adv_text = random.choice(PREAMBLES).format(text=fact_text)
            
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "target_axis": "advice_intensity",
                "provenance": "genuine_regulatory",
                "generation_strategy": "within_domain",
                "known_confounds": [],
                "target_text": f"{prompt_prefix}{adv_text}",
                "noise_text": f"{prompt_prefix}{fact_text}",
                "metrics": {
                    "jaccard_similarity": 0.5, 
                    "semantic_cosine": 0.5     
                }
            })
            
    corpus = {
        "corpus_id": "rg36_v2_high_variance_afca",
        "target_manifold": "ASIC RG 36 Personal Advice (High Variance AFCA)",
        "confound_registry": {},
        "axis_roles": {
            "primary_structural": "advice_intensity",
            "secondary_behavioural": "client_focus" 
        },
        "pairs": pairs
    }
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Bridged {len(pairs)} high-variance pairs. Saving to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rg36_corpus/pairs/afca_only_pairs.jsonl")
    parser.add_argument("--output", default="configs/rg36_v2_bridged_corpus.json")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    args = parser.parse_args()
    bridge_corpus(args.input, args.output, args.model)
