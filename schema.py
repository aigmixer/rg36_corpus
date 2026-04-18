from dataclasses import dataclass, field
from typing import Literal, List, Optional

Label = Literal["FACTUAL", "GENERAL", "PERSONAL", "BOUNDARY"]
Source = Literal["AFCA", "ASIC", "RBA", "MONEYSMART", "SYNTHETIC"]

@dataclass
class ContrastivePair:
    pair_id: str
    vector_a: str                    # Lower-risk text (FACTUAL/GENERAL)
    label_a: Label = "FACTUAL"
    vector_b: str = ""               # Higher-risk text (GENERAL/PERSONAL)  
    label_b: Label = "PERSONAL"
    source: Source = "SYNTHETIC"
    
    # Statutory markers
    has_second_person_possessive: bool = False
    has_personal_variable: bool = False
    has_westpac_pattern: bool = False
    
    # Quality scores  
    boundary_precision: float = 0.0
    legal_grounding: float = 0.0
    syntactic_delta: str = ""
    
    # Metadata for attribution
    source_doc: str = ""
    entry_ref: str = ""
    metadata: dict = field(default_factory=dict)

@dataclass
class ParsedDocument:
    doc_id: str
    source: Source
    title: str
    url: str
    sentences: List[str]
    metadata: dict = field(default_factory=dict)
