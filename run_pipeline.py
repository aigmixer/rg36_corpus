import argparse
import json
import os
from typing import List
from tqdm import tqdm
from rg36_corpus.schema import ContrastivePair
from rg36_corpus.ingestion_pipeline import run_ingestion
from rg36_corpus.pair_generator import strip_pairs_from_doc, generate_inject_pairs, boundary_pairs_from_afca
from rg36_corpus.pair_auditor import run_audit

def save_jsonl(data: List, path: str):
    """Save a list of objects to JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            if hasattr(item, "__dict__"):
                f.write(json.dumps(item.__dict__, default=str) + "\n")
            else:
                f.write(json.dumps(item, default=str) + "\n")

def main():
    parser = argparse.ArgumentParser(description="RG 36 Corpus Engineering Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (10 synthetic pairs)")
    parser.add_argument("--sources", nargs="+", default=["rba", "asic", "moneysmart"], help="Data sources to ingest")
    parser.add_argument("--target", type=int, default=10000, help="Target number of pairs")
    parser.add_argument("--pairs-only", action="store_true", help="Skip ingestion, use existing parsed docs (mocked)")
    parser.add_argument("--audit-only", action="store_true", help="Only audit existing pairs")
    
    args = parser.parse_args()
    
    pairs: List[ContrastivePair] = []
    
    if args.demo:
        print("Running in DEMO mode...")
        pairs = generate_inject_pairs(target_count=10)
    elif args.audit_only:
        print("Running AUDIT only on existing pairs...")
        # (Implementation would load from pairs/rg36_pairs.jsonl)
        print("Feature not fully implemented in this version.")
        return
    else:
        # 1. Ingestion
        print(f"Ingesting from sources: {args.sources}")
        docs = run_ingestion(args.sources)
        
        # 2. Pair Generation
        print("Generating pairs...")
        
        # Strategy 1 & 3 from ingested docs
        for doc in tqdm(docs, desc="Generating from docs"):
            pairs.extend(strip_pairs_from_doc(doc))
            if doc.source == "AFCA":
                pairs.extend(boundary_pairs_from_afca(doc))
        
        # Strategy 4: Hybrid Injection (Real Factual -> Synthetic Personal)
        # We target a significant portion of 'Real' content here
        hybrid_target = (args.target // 2) - len(pairs)
        if hybrid_target > 0:
            print(f"Generating {hybrid_target} hybrid pairs from real factual text...")
            from .pair_generator import generate_hybrid_pairs
            pairs.extend(generate_hybrid_pairs(docs, hybrid_target))

        # Strategy 2: Fill the rest with synthetic pairs
        remaining = args.target - len(pairs)
        if remaining > 0:
            print(f"Filling remaining {remaining} pairs with synthetic templates...")
            # We update generate_inject_pairs internally or just use a loop here for tqdm
            # For simplicity, we'll wrap the generation in a tqdm loop if target is large
            batch_size = 100
            for _ in tqdm(range(0, remaining, batch_size), desc="Synthetic Generation"):
                count = min(batch_size, remaining - len(pairs) + remaining % batch_size if remaining < batch_size else batch_size)
                # Note: len(pairs) is used to track progress
                if len(pairs) >= args.target: break
                pairs.extend(generate_inject_pairs(target_count=count))
            
    # 3. Audit
    print(f"Auditing {len(pairs)} pairs...")
    # Update run_audit to support tqdm or wrap it here
    audit_results = run_audit(pairs)
    
    # 4. Save results
    print("Saving results...")
    save_jsonl(pairs, "rg36_corpus/pairs/rg36_pairs.jsonl")
    save_jsonl(audit_results, "rg36_corpus/audit/audit_results.jsonl")
    
    pass_count = sum(1 for r in audit_results if r.passed)
    summary = {
        "total_pairs": len(pairs),
        "passed": pass_count,
        "pass_rate": pass_count / len(pairs) if pairs else 0,
        "sources": args.sources if not args.demo else ["DEMO"]
    }
    
    with open("rg36_corpus/audit/audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nPipeline Complete.")
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    print(f"Results saved to rg36_corpus/pairs/rg36_pairs.jsonl")

if __name__ == "__main__":
    main()
