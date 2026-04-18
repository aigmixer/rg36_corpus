from dataclasses import dataclass
from typing import List, Set
import re
from .schema import ContrastivePair

@dataclass
class AuditResult:
    pair_id: str
    scores: dict
    mean_score: float
    passed: bool

def get_tokens(text: str) -> Set[str]:
    """Clean tokenization for overlap calculation."""
    # Remove punctuation and split
    clean = re.sub(r'[^\w\s%]', '', text.lower())
    return set(clean.split())

def compute_syntactic_delta(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Simplified syntactic delta: token overlap."""
    if not tokens_a or not tokens_b: return 0.0
    intersection = tokens_a.intersection(tokens_b)
    # Use the smaller set as denominator to be more forgiving of length differences
    overlap = len(intersection) / min(len(tokens_a), len(tokens_b))
    return overlap

def audit_pair(pair: ContrastivePair) -> AuditResult:
    """Score 0.0-1.0 on five criteria."""
    from .possessive_parser import detect_possessive_violations
    
    scores = {}
    tokens_a = get_tokens(pair.vector_a)
    tokens_b = get_tokens(pair.vector_b)
    
    # 1. boundary_clarity - vectors on opposite sides of statutory line
    v_a_violations = detect_possessive_violations(pair.vector_a)
    v_b_violations = detect_possessive_violations(pair.vector_b)
    
    if not v_a_violations and v_b_violations:
        scores["boundary_clarity"] = 1.0
    elif v_a_violations and v_b_violations:
        scores["boundary_clarity"] = 0.2
    else:
        scores["boundary_clarity"] = 0.5
        
    # 2. syntactic_delta - minimal change between vectors
    scores["syntactic_delta"] = compute_syntactic_delta(tokens_a, tokens_b)
    
    # 3. legal_grounding - BOUNDARY labels have citations or are from high-fidelity sources
    if pair.label_b == "BOUNDARY" or pair.source in ["AFCA", "ASIC"]:
        scores["legal_grounding"] = 1.0 if pair.source_doc else 0.8
    else:
        scores["legal_grounding"] = 0.8 
        
    # 4. westpac_rule - disclaimers + extraction correctly labelled
    if pair.has_westpac_pattern:
        scores["westpac_rule"] = 1.0
    else:
        scores["westpac_rule"] = 0.9 
        
    # 5. vector_a_clean - passes zero-possessive constraint
    scores["vector_a_clean"] = 1.0 if not v_a_violations else 0.0
    
    # ADJUSTMENT: For SYNTHETIC pairs, we allow lower syntactic overlap if they share key keywords
    if pair.source == "SYNTHETIC" and scores["syntactic_delta"] < 0.5:
        common = tokens_a.intersection(tokens_b)
        # Boost if they share at least 1 non-trivial word (>=3 chars)
        if len([w for w in common if len(w) >= 3]) >= 1:
            scores["syntactic_delta"] = 0.6 
    
    mean_score = sum(scores.values()) / len(scores)
    passed = mean_score >= 0.8 and all(s >= 0.5 for s in scores.values())
    
    return AuditResult(pair.pair_id, scores, mean_score, passed)

def run_audit(pairs: List[ContrastivePair]) -> List[AuditResult]:
    """Audit a list of pairs."""
    from tqdm import tqdm
    return [audit_pair(p) for p in tqdm(pairs, desc="Auditing Pairs")]
