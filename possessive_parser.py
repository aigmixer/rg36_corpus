import re
from typing import List

# List of financial state variables that trigger violations when combined with second-person possessives
FINANCIAL_STATE_VARS = [
    "super", "superannuation", "balance", "salary", "income", "mortgage", "loan", 
    "repayments", "risk profile", "investment horizon", "objectives", "needs", 
    "circumstances", "portfolio", "savings", "debt", "budget", "expenses"
]

SECOND_PERSON_POSSESSIVES = ["your", "you're", "you are", "yours"]

def detect_possessive_violations(text: str) -> List[str]:
    """Find spans where second-person possessive + financial state variable"""
    violations = []
    text_lower = text.lower()
    
    for possessive in SECOND_PERSON_POSSESSIVES:
        for var in FINANCIAL_STATE_VARS:
            pattern = rf"\b{possessive}\s+(?:\w+\s+)?{var}\b"
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                violations.append(text[match.start():match.end()])
    
    return violations

def detect_westpac_pattern(text: str) -> bool:
    """Disclaimer followed by personal objective extraction.
    Rule: Disclaimer ('general advice only') followed by consideration of personal factors.
    """
    text_lower = text.lower()
    has_disclaimer = any(phrase in text_lower for phrase in ["general advice", "disclaimer", "not consider", "personal circumstances"])
    has_personal = any(phrase in text_lower for phrase in ["given your", "considering your", "based on your", "because you have"])
    
    return has_disclaimer and has_personal

def strip_possessive_framing(text: str) -> str:
    """24 transformation rules: 'your super' -> 'the member's super'
    Isolates semantic intent from personal framing.
    """
    transformations = [
        (r"\b[Yy]our superannuation\b", "the member's superannuation"),
        (r"\b[Yy]our super\b", "the member's super"),
        (r"\b[Yy]our balance\b", "the account balance"),
        (r"\b[Yy]our salary\b", "the individual's salary"),
        (r"\b[Yy]our income\b", "the individual's income"),
        (r"\b[Yy]our mortgage\b", "a mortgage"),
        (r"\b[Yy]our loan\b", "a loan"),
        (r"\b[Yy]our repayments\b", "the repayments"),
        (r"\b[Yy]our risk profile\b", "a typical risk profile"),
        (r"\b[Yy]our investment horizon\b", "the investment horizon"),
        (r"\b[Yy]our objectives\b", "the stated objectives"),
        (r"\b[Yy]our needs\b", "the relevant needs"),
        (r"\b[Yy]our circumstances\b", "the relevant circumstances"),
        (r"\b[Yy]our portfolio\b", "a portfolio"),
        (r"\b[Yy]our savings\b", "the accumulated savings"),
        (r"\b[Yy]our debt\b", "the outstanding debt"),
        (r"\b[Yy]our budget\b", "the budget"),
        (r"\b[Yy]our expenses\b", "the expenses"),
        (r"\b[Yy]ou should\b", "it may be appropriate to"),
        (r"\b[Yy]ou could\b", "it is possible to"),
        (r"\b[Yy]ou need to\b", "it is necessary to"),
        (r"\b[Gg]iven your\b", "In cases of"),
        (r"\b[Cc]onsidering your\b", "When considering"),
        (r"\b[Bb]ased on your\b", "Based on"),
    ]
    
    stripped_text = text
    for pattern, replacement in transformations:
        stripped_text = re.sub(pattern, replacement, stripped_text)
    
    return stripped_text
