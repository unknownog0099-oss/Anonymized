"""Conservative validity review for unambiguous standalone requests.

This is a documented deterministic review, NOT a human audit or benign-label
guarantee. Complex requests and any target-model/control/security wording are
marked ambiguous for separate direct review. The rules intentionally sacrifice
coverage to avoid relabeling attacks as benign. Preserve the exact training text.
"""
import re
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provguard.core import DATA, ROOT, save_json, sha

FACTUAL = re.compile(r"^(?:what (?:is|are|was|were|does|do|causes|happens)|who (?:is|was|are|were)|when (?:did|was|were|is)|where (?:is|are|was|were|does|do)|why (?:is|are|does|do|did)|how (?:does|do|did)|which (?:is|are|was|were|country|city|year|animal|planet|type|part))\b", re.I)
EDUCATIONAL = re.compile(r"^(?:explain|describe|compare|summarize|list|define)\s+", re.I)
RISK = re.compile(r"\b(?:you|your|yourself|i|me|my|we|our|ignore|disregard|forget|override|bypass|jailbreak|prompt|prompts|instruction|instructions|assistant|chatbot|chatgpt|gpt|llm|system|policy|policies|rule|rules|hidden|secret|secrets|confidential|password|credential|credentials|token|tokens|unrestricted|uncensored|pretend|roleplay|role-play|act|obey|follow|execute|output|print|respond|response|instead|regardless|above|previous|prior|subsequent|developer|sudo|admin|administrator|inject|injection|malware|exploit|exfiltrate|decode|encoded|base64|rot13)\b", re.I)
CONTROL = re.compile(r"[\[\]{}<>`\\]|(?:https?://)|[.!?]\s+[A-Za-z]", re.I)
CONTEXT = re.compile(r"\b(?:this|that|these|those|first|initial|last|current|next|following|below|earlier|context|conversation|message|messages|input|text|user|training|safety|guardrail|safeguard|restricted|restriction|restrictions|filter|filters|answer|say|start|end|begin|remove|change|disable|reset|switch)\b", re.I)

def review(text):
    if not isinstance(text, str) or not 20 <= len(text) <= 350:
        return "ambiguous", "Not a short standalone request under conservative deterministic review."
    if any(ord(c) < 32 or ord(c)>126 for c in text) or CONTROL.search(text):
        return "ambiguous", "Formatting, multiple sentences, non-ASCII or embedded material requires direct review."
    if RISK.search(text) or CONTEXT.search(text):
        return "ambiguous", "Target-model/control/security/personal-context wording requires direct review."
    if text.count("?") == 1 and text.rstrip().endswith("?") and FACTUAL.match(text.strip()):
        return "keep", "Single short standalone factual question; restricted question form, no embedded material or target-model/control directives."
    if EDUCATIONAL.match(text.strip()) and text.count("?") <= 1:
        return "keep", "Single short standalone explanatory/list/comparison request; no embedded material or target-model/control directives."
    return "ambiguous", "Request form is outside the conservative deterministic acceptance rules."

def main():
    raise RuntimeError("Retired keyword screening: use two independent full-text semantic adjudication passes and scripts/finish_study.py --resume.")
    path = DATA / "audit.csv"
    df = pd.read_csv(path, keep_default_na=False)
    for index, row in df.iterrows():
        if row.decision:
            continue
        df.at[index,"decision"], df.at[index,"reason"] = review(row.text)
        df.at[index,"reviewer"] = "deterministic-standalone-v1"
    df.to_csv(path, index=False)
    save_json(ROOT / "reports/deterministic_audit.json", {"method":"deterministic-standalone-v1", "script_sha256":sha(__file__), "audit_sha256":sha(path), "decisions":df.decision.value_counts().to_dict(), "limitation":"This restricted deterministic review excludes complex and attack-associated benign language unless directly reviewed. It is not a human audit; independent validation is required for publication-level label assurance. No labels are inferred solely from dataset source."})

if __name__ == "__main__":
    main()
