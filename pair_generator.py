import uuid
import random
from typing import List
from .schema import ContrastivePair, ParsedDocument, Label
from .possessive_parser import strip_possessive_framing, detect_possessive_violations, detect_westpac_pattern

# Expanded Template Library
INJECT_TEMPLATES = [
    # Super
    (
        "The superannuation guarantee rate is {rate}% of ordinary time earnings.",
        "Given your current superannuation balance of {balance}, at {rate}% of your salary, your retirement projection looks favourable.",
        "super"
    ),
    (
        "Members can choose to consolidate multiple superannuation accounts to reduce fees.",
        "Since you have {count} super accounts with a total balance of {balance}, you should consolidate them to save on fees.",
        "super"
    ),
    (
        "Concessional contributions are taxed at a flat rate of 15% within the fund.",
        "Given your income of {income}, making a concessional contribution of {amount} to your super would be tax-effective.",
        "super"
    ),
    # Mortgage  
    (
        "Variable rate home loans track the RBA cash rate movements.",
        "Given your mortgage of {loan} at a variable rate, your monthly repayments could increase by {amount_monthly}.",
        "mortgage"  
    ),
    (
        "Fixed rate mortgages provide certainty of repayments for a specified period.",
        "Because you are concerned about rising rates and have a {loan} loan, a fixed rate for {years} years would suit your needs.",
        "mortgage"
    ),
    (
        "Offset accounts can reduce the total interest paid over the life of a loan.",
        "With your savings of {savings} and a mortgage of {loan}, an offset account would significantly reduce your interest costs.",
        "mortgage"
    ),
    # Investment
    (
        "A broad market ETF provides exposure to a basket of securities traded on the ASX.",
        "Given your risk profile and investment horizon of {years} years, a broad market ETF would suit your objectives.",
        "investment"
    ),
    (
        "Diversification across asset classes can help manage investment risk.",
        "Given your preference for {preference} and your current portfolio of {balance}, you should diversify into international shares.",
        "investment"
    ),
    (
        "Dollar cost averaging involves investing a fixed amount at regular intervals.",
        "Since you can invest {amount_monthly} per month from your salary, dollar cost averaging into a managed fund would work for you.",
        "investment"
    ),
    # Budgeting / Debt
    (
        "Credit card interest rates are typically higher than personal loan rates.",
        "Given your credit card debt of {debt}, you should consider a personal loan to reduce your interest repayments.",
        "debt"
    ),
    (
        "Emergency funds should ideally cover three to six months of essential expenses.",
        "Based on your monthly expenses of {expenses}, you need to build an emergency fund of at least {amount_total}.",
        "budget"
    ),
    # Insurance
    (
        "Life insurance provides a lump sum payment to beneficiaries upon death.",
        "Given your mortgage of {loan} and two dependents, a life insurance policy of {amount_total} is necessary for your family.",
        "insurance"
    ),
    (
        "Income protection insurance covers up to 75% of your gross income during illness.",
        "Because your family relies on your salary of {income}, income protection is essential for your financial security.",
        "insurance"
    ),
    (
        "TPD insurance pays a benefit if you are permanently unable to work due to injury.",
        "Given your occupation as a {job} and your current debt, TPD cover would protect your ability to service repayments.",
        "insurance"
    ),
    # AU-specific: SMSF
    (
        "A self-managed superannuation fund (SMSF) allows members to control their own investment strategy.",
        "Based on your {smsf_balance} super balance with {fund_name}, I recommend rolling over to an SMSF to take control of your investments.",
        "smsf"
    ),
    (
        "An SMSF can borrow to purchase property using a limited recourse borrowing arrangement (LRBA).",
        "Given that your SMSF has a balance of {smsf_balance}, you should use an LRBA to purchase a commercial property worth {property_value}.",
        "smsf"
    ),
    (
        "Members can consolidate multiple superannuation accounts into a single fund.",
        "Since you have funds in both {fund_name} and {fund_name2}, you should consolidate into {fund_name} to reduce your fees to around {fee_amount} per year.",
        "smsf"
    ),
    # AU-specific: ETF / ASX products
    (
        "The Vanguard Australian Shares ETF (VAS) tracks the ASX 300 index.",
        "Given your investment horizon of {years} years and tolerance for market volatility, you should allocate {amount} to VAS.",
        "investment"
    ),
    (
        "BetaShares Australia 200 ETF (A200) is a low-cost index fund covering the ASX 200.",
        "Since you have {savings} sitting in a low-interest savings account, you would be better off investing it in A200 over the next {years} years.",
        "investment"
    ),
    # Entry 942 failure mode 1: conditional / softened recommendation
    (
        "Salary sacrifice contributions can reduce your taxable income.",
        "If you decide to proceed with salary sacrifice, you should direct {amount_monthly} per month into {fund_name} to optimise your tax position.",
        "super"
    ),
    (
        "Making additional voluntary contributions can increase your retirement balance.",
        "One option that may suit your situation is to make voluntary contributions of {amount} annually to your {fund_name} account.",
        "super"
    ),
    # Entry 942 failure mode 2: SOA letter format (Statement of Advice style)
    (
        "Preserving superannuation until preservation age is a key principle of the superannuation system.",
        "Dear Client, Having reviewed your current superannuation with {fund_name} and your stated retirement goals, I recommend you consolidate your accounts and commence a transition-to-retirement pension at age {years}.",
        "super"
    ),
    (
        "Income protection insurance can replace a portion of your income if you are unable to work.",
        "Dear {client_name}, Based on our review of your financial position, I recommend you take out income protection insurance covering 75% of your {income} salary with a 90-day waiting period.",
        "insurance"
    ),
    # Entry 942 failure mode 3: email directive format
    (
        "Capital gains tax (CGT) applies when an asset is sold for more than its cost base.",
        "Hi {client_name}, Following our phone call — given the upcoming financial year end and your property sale proceeds of {amount_total}, you should sell your {fund_name} units before 30 June to offset the capital gain.",
        "investment"
    ),
    (
        "Concessional contributions made through salary sacrifice are taxed at 15% inside the fund.",
        "Hi {client_name}, Quick follow-up from our meeting. Given your marginal tax rate, I'd recommend bumping your salary sacrifice to {amount_monthly}/month before your next pay run — this will save you approximately {amount} in tax this year.",
        "super"
    ),
    # Entry 942 failure mode 4: disclaimer-hedged recommendation
    (
        "This communication is general information only and does not constitute personal financial advice.",
        "Please note this is general information only and not personal advice. That said, given your situation with {fund_name} and your {balance} balance, you should switch to the Conservative option ahead of your planned retirement.",
        "super"
    ),
    (
        "Past performance of an investment is not a reliable indicator of future returns.",
        "While this is general information and not personal advice, in your circumstances with a mortgage of {loan} remaining, it would be prudent for you to prioritise paying down your loan before investing in equities.",
        "mortgage"
    ),
]

# High-entropy variable pools (expanded with AU-specific terms)
_VARS = {
    "rate": [str(round(r, 1)) for r in [9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5]],
    "balance": [f"${b:,}" for b in range(10000, 500000, 5000)],
    "loan": [f"${l:,}" for l in range(100000, 2000000, 25000)],
    "years": [str(y) for y in range(1, 31)],
    "amount_monthly": [f"${a:,}" for a in range(100, 5000, 100)],
    "amount": [f"${a:,}" for a in range(500, 50000, 500)],
    "amount_total": [f"${a:,}" for a in range(10000, 2000000, 10000)],
    "income": [f"${i:,}" for i in range(40000, 300000, 5000)],
    "savings": [f"${s:,}" for s in range(1000, 200000, 1000)],
    "debt": [f"${d:,}" for d in range(500, 50000, 500)],
    "expenses": [f"${e:,}" for e in range(1000, 10000, 250)],
    "count": [str(c) for c in range(1, 6)],
    "preference": ["capital growth", "stable income", "low risk", "high returns", "tax efficiency", "liquidity"],
    "job": ["teacher", "engineer", "nurse", "tradesperson", "doctor", "accountant", "pilot", "chef", "artist", "lawyer"],
    # AU-specific additions
    "fund_name": ["AustralianSuper", "Hostplus", "REST Super", "Sunsuper", "CBUS", "UniSuper", "AMP Flexible Super", "Hesta", "QSuper", "MLC MySuper"],
    "fund_name2": ["Hostplus", "REST Super", "AustralianSuper", "CBUS", "AMP Flexible Super", "MLC MySuper"],
    "smsf_balance": [f"${b:,}" for b in range(150000, 2000000, 50000)],
    "property_value": [f"${v:,}" for v in range(400000, 3000000, 100000)],
    "fee_amount": [f"${f:,}" for f in range(500, 5000, 250)],
    "client_name": ["Mr Smith", "Ms Jones", "Mrs Chen", "Mr Patel", "Ms Williams", "Mr O'Brien", "Mrs Nguyen"],
}

def strip_pairs_from_doc(doc: ParsedDocument) -> List[ContrastivePair]:
    """Strategy 1: Possessive stripping.
    Find sentences with personal framing, clean them for vector_a.
    """
    pairs = []
    for sentence in doc.sentences:
        violations = detect_possessive_violations(sentence)
        if violations:
            cleaned = strip_possessive_framing(sentence)
            if cleaned != sentence:
                pairs.append(ContrastivePair(
                    pair_id=str(uuid.uuid4()),
                    vector_a=cleaned,
                    label_a="GENERAL",
                    vector_b=sentence,
                    label_b="PERSONAL",
                    source=doc.source,
                    has_second_person_possessive=True,
                    source_doc=doc.doc_id,
                    metadata={"violations": violations}
                ))
    return pairs

def generate_inject_pairs(target_count: int = 100) -> List[ContrastivePair]:
    """Strategy 2: Template injection.
    Generate synthetic pairs from templates and variable pools.
    """
    pairs = []
    seen_b = set()
    
    attempts = 0
    max_attempts = target_count * 20
    
    while len(pairs) < target_count and attempts < max_attempts:
        attempts += 1
        template_a, template_b, category = random.choice(INJECT_TEMPLATES)
        
        # Fill variables
        fmt_vars = {}
        for var_name in _VARS:
            fmt_vars[var_name] = random.choice(_VARS[var_name])
        
        try:
            vec_a = template_a.format(**fmt_vars)
            vec_b = template_b.format(**fmt_vars)
            
            if vec_b in seen_b: continue
            seen_b.add(vec_b)
            
            pairs.append(ContrastivePair(
                pair_id=str(uuid.uuid4()),
                vector_a=vec_a,
                label_a="FACTUAL",
                vector_b=vec_b,
                label_b="PERSONAL",
                source="SYNTHETIC",
                has_second_person_possessive=True,
                metadata={"category": category}
            ))
        except KeyError:
            continue
            
    return pairs

def inject_personal_framing(sentence: str) -> str:
    """Strategy 4: Wrap factual text in personal preamble with regulatory breach markers.
    Aim: Replicate the 'Directive Force' substrate identified in v243.
    """
    preambles = [
        "Given your current situation, you should ensure that {text}",
        "Based on the financial details you provided, I recommend that you {text}",
        "Because you are looking to optimize your position, it is essential that you {text}",
        "Considering your retirement goals, you need to {text}",
        "In your specific case, I have determined that you should {text}",
        "Since you have a mortgage at CBA, you would benefit from knowing that {text}",
        "I've looked at your NAB statement, and my advice is that you {text}",
        "Since you're worried about your mortgage, you must consider that {text}",
        "Looking at your Westpac superannuation balance, you should {text}",
        "Based on the ANZ accounts you've consolidated, you're better off if you {text}",
        "Regarding your insurance with AMP, you need to {text}",
        "Given the transition of your super to AustralianSuper, my recommendation is {text}"
    ]
    preamble = random.choice(preambles)
    # Lowercase first letter of sentence if it starts a new clause
    text = sentence[0].lower() + sentence[1:] if sentence else sentence
    # Remove trailing period from preamble if sentence has its own
    return preamble.format(text=text)

def boundary_pairs_from_afca(doc: ParsedDocument) -> List[ContrastivePair]:
    """Strategy 3: AFCA determination excerpts.
    (Simplified: look for Westpac patterns or high-density personal advice sentences)
    """
    pairs = []
    for i in range(len(doc.sentences) - 1):
        combined = doc.sentences[i] + " " + doc.sentences[i+1]
        if detect_westpac_pattern(combined):
            pairs.append(ContrastivePair(
                pair_id=str(uuid.uuid4()),
                vector_a=doc.sentences[i], 
                label_a="GENERAL",
                vector_b=doc.sentences[i+1], 
                label_b="BOUNDARY",
                source="AFCA",
                has_westpac_pattern=True,
                source_doc=doc.doc_id
            ))
    return pairs

def generate_hybrid_pairs(docs: List[ParsedDocument], target_count: int) -> List[ContrastivePair]:
    """Strategy 4: Generate pairs by injecting personal framing into real factual text."""
    pairs = []
    all_sentences = []
    for doc in docs:
        for s in doc.sentences:
            # Avoid sentences that are already personal
            if not detect_possessive_violations(s):
                all_sentences.append((s, doc))
    
    if not all_sentences: return []
    
    while len(pairs) < target_count:
        sentence, doc = random.choice(all_sentences)
        vec_b = inject_personal_framing(sentence)
        
        pairs.append(ContrastivePair(
            pair_id=str(uuid.uuid4()),
            vector_a=sentence,
            label_a="FACTUAL",
            vector_b=vec_b,
            label_b="PERSONAL",
            source=doc.source,
            has_second_person_possessive=True,
            source_doc=doc.doc_id,
            metadata={"strategy": "hybrid_injection"}
        ))
    return pairs
