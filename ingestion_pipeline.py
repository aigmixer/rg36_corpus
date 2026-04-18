import os
import requests
import time
import pdfplumber
import io
import glob
from bs4 import BeautifulSoup
from typing import List, Optional
from tqdm import tqdm
from .schema import ParsedDocument, Source

# Server Etiquette Configuration
HEADERS = {
    "User-Agent": "Academic Research Bot/1.0 (Financial Regulation Corpus; contact@university.edu)"
}

# Direct Verified PDF Links
ASIC_PDF_URLS = [
    "https://download.asic.gov.au/media/wdnk4aja/rg36-published-8-june-2016-20220328.pdf",
    "https://download.asic.gov.au/media/pqpe0hwc/rg175-published-21-november-2024-20241219.pdf",
    "https://download.asic.gov.au/media/4531113/rg244-published-13-december-2012.pdf",
    "https://download.asic.gov.au/media/etgm1amc/rg274-published-10-september-2024.pdf",
    "https://download.asic.gov.au/media/pdvbtvqr/rg271-published-2-september-2021.pdf",
    "https://download.asic.gov.au/media/5411131/rg38-published-20-december-2019.pdf",
    "https://download.asic.gov.au/media/3013754/rg146-published-2-july-2012-20141218.pdf",
    "https://download.asic.gov.au/media/v0vhrun4/rg1-published-25-july-2023.pdf",
    # New verified direct links
    "https://asic.gov.au/regulatory-resources/find-a-document/regulatory-guides/rg-104-afs-licensing-meeting-the-general-obligations/RG-104-AFS-licensing-Meeting-the-general-obligations.pdf",
    "https://asic.gov.au/regulatory-resources/find-a-document/regulatory-guides/rg-105-afs-licensing-organisational-competence/RG-105-AFS-licensing-Organisational-competence.pdf",
    "https://asic.gov.au/regulatory-resources/find-a-document/regulatory-guides/rg-166-afs-licensing-financial-requirements/RG-166-AFS-licensing-Financial-requirements.pdf",
    "https://asic.gov.au/regulatory-resources/find-a-document/regulatory-guides/rg-181-afs-licensing-managing-conflicts-of-interest/RG-181-AFS-licensing-Managing-conflicts-of-interest.pdf",
    "https://asic.gov.au/regulatory-resources/find-a-document/regulatory-guides/rg-246-conflicted-and-other-banned-remuneration/RG-246-Conflicted-and-other-banned-remuneration.pdf"
]

MONEYSMART_URLS = [
    "https://moneysmart.gov.au/how-super-works/what-is-superannuation",
    "https://moneysmart.gov.au/how-super-works/choosing-a-super-fund",
    "https://moneysmart.gov.au/grow-your-super/super-investment-options",
    "https://moneysmart.gov.au/how-super-works/superannuation-calculator",
    "https://moneysmart.gov.au/managed-funds-and-etfs/exchange-traded-funds-etfs",
    "https://moneysmart.gov.au/home-loans/choosing-a-home-loan",
    "https://moneysmart.gov.au/investing/investing-basics",
    "https://moneysmart.gov.au/banking/savings-accounts",
    "https://moneysmart.gov.au/insurance/life-insurance",
    "https://moneysmart.gov.au/insurance/income-protection-insurance",
    "https://moneysmart.gov.au/retirement-income/planning-your-retirement",
    "https://moneysmart.gov.au/investing/investing-risk",
    "https://moneysmart.gov.au/investing/choosing-investments-to-suit-you"
]

def fetch_with_backoff(url, max_retries=3):
    """Exponential backoff for respectful fetching."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch {url}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None

def ingest_asic_pdfs() -> List[ParsedDocument]:
    """Fetch and parse ASIC Regulatory Guides."""
    docs = []
    for url in tqdm(ASIC_PDF_URLS, desc="Ingesting ASIC PDFs"):
        response = fetch_with_backoff(url)
        if not response: continue
        
        # Save raw PDF
        filename = os.path.basename(url)
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        raw_dir = "rg36_corpus/raw/asic"
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, filename)
        with open(raw_path, "wb") as f:
            f.write(response.content)
        
        # Parse PDF
        sentences = []
        try:
            with pdfplumber.open(raw_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        # Basic sentence splitting
                        page_sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
                        sentences.extend(page_sentences)
        except Exception as e:
            print(f"Error parsing PDF {filename}: {e}")
        
        docs.append(ParsedDocument(
            doc_id=filename,
            source="ASIC",
            title=f"Regulatory Guide: {filename}",
            url=url,
            sentences=sentences
        ))
        time.sleep(2) # Throttle
    return docs

def ingest_moneysmart() -> List[ParsedDocument]:
    """Fetch and parse Moneysmart HTML pages."""
    docs = []
    for url in tqdm(MONEYSMART_URLS, desc="Ingesting Moneysmart"):
        response = fetch_with_backoff(url)
        if not response: continue
        
        # Save raw HTML
        doc_id = url.split("/")[-1]
        raw_dir = "rg36_corpus/raw/moneysmart"
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, f"{doc_id}.html")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        
        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        content = soup.find("main") or soup.find("article") or soup.body
        if not content: continue
        
        paragraphs = content.find_all("p")
        sentences = []
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 20:
                sentences.extend([s.strip() for s in text.split(".") if len(s.strip()) > 20])
        
        docs.append(ParsedDocument(
            doc_id=doc_id,
            source="MONEYSMART",
            title=soup.title.string if soup.title else doc_id,
            url=url,
            sentences=sentences
        ))
        time.sleep(2) # Throttle
    return docs

def ingest_rba_local() -> List[ParsedDocument]:
    """Read search-extracted RBA minutes from local text files."""
    docs = []
    rba_files = glob.glob("rg36_corpus/raw/rba/*.txt")
    for file_path in tqdm(rba_files, desc="Ingesting RBA Local"):
        with open(file_path, "r") as f:
            text = f.read()
            
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        doc_id = os.path.basename(file_path)
        docs.append(ParsedDocument(
            doc_id=doc_id,
            source="RBA",
            title=f"RBA Minute: {doc_id}",
            url="local",
            sentences=sentences
        ))
    return docs

def ingest_afca_local() -> List[ParsedDocument]:
    """Parse manually downloaded AFCA determination PDFs."""
    docs = []
    afca_files = glob.glob("rg36_corpus/raw/afca/*.pdf")
    for raw_path in tqdm(afca_files, desc="Ingesting AFCA PDFs"):
        filename = os.path.basename(raw_path)
        sentences = []
        try:
            with pdfplumber.open(raw_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        page_sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
                        sentences.extend(page_sentences)
        except Exception as e:
            print(f"Error parsing AFCA PDF {filename}: {e}")
            continue
            
        docs.append(ParsedDocument(
            doc_id=filename,
            source="AFCA",
            title=f"AFCA Determination: {filename}",
            url="local",
            sentences=sentences
        ))
    return docs

def run_ingestion(sources: List[str]) -> List[ParsedDocument]:
    """Orchestrate ingestion from multiple sources."""
    all_docs = []
    if "asic" in sources:
        all_docs.extend(ingest_asic_pdfs())
    if "moneysmart" in sources:
        all_docs.extend(ingest_moneysmart())
    if "rba" in sources:
        all_docs.extend(ingest_rba_local())
    if "afca" in sources:
        all_docs.extend(ingest_afca_local())
    
    return all_docs
