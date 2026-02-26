"""Create a lexical control corpus to test feature robustness.

The reviewer raised a construct validity concern: 
"Possible conflation of lexical category differences (e.g., 'penalty', 'grant') 
with general safety vs innovation cognition... Can you rule out lexical confounds?"

This script takes the D_safe and D_innov datasets and replaces primary structural 
keywords with synonyms or masked terms. We will then extract activations on this 
control corpus to see if the 42 Layer 24 features still polarize on the abstract 
concept rather than mere lexical triggers.
"""

import json
import re
from pathlib import Path

# Dictionary of lexical confound replacements
# Format: {pattern: replacement}
REPLACEMENTS = {
    # D_safe (Restrictive) keywords
    r"\bpenalty\b": "consequence",
    r"\bpenalties\b": "consequences",
    r"\bfine\b": "charge",
    r"\bfines\b": "charges",
    r"\bprohibit\b": "restrict",
    r"\bprohibited\b": "restricted",
    r"\bshall not\b": "must avoid",
    r"\baudit\b": "review",
    r"\baudits\b": "reviews",
    r"\bliable\b": "responsible",
    r"\bliability\b": "responsibility",
    r"\bviolation\b": "infraction",
    r"\bban\b": "block",
    r"\bbanned\b": "blocked",
    r"\brestriction\b": "limitation",
    r"\bcompliance\b": "adherence",

    # D_innov (Incentive) keywords
    r"\bgrant\b": "funding allocation",
    r"\bgrants\b": "funding allocations",
    r"\bcredit\b": "benefit",
    r"\bcredits\b": "benefits",
    r"\btax\b": "fiscal",
    r"\bincentive\b": "encouragement",
    r"\bincentives\b": "encouragements",
    r"\bfund\b": "support pool",
    r"\bfunding\b": "financial support",
    r"\bexempt\b": "excused",
    r"\bexemption\b": "exception",
    r"\bderegulation\b": "policy easing",
    r"\binnovation\b": "advancement",
    r"\bresearch and development\b": "exploratory work",
    r"\br&d\b": "exploratory work",
}

def apply_lexical_control(text: str) -> str:
    """Apply regex replacements (case-insensitive) to control for lexical cues."""
    controlled = text
    for pattern, repl in REPLACEMENTS.items():
        controlled = re.sub(pattern, repl, controlled, flags=re.IGNORECASE)
    return controlled

def process_file(in_path: Path, out_path: Path):
    print(f"Processing {in_path.name} -> {out_path.name}")
    lines = in_path.read_text("utf-8").strip().split("\n")
    processed_count = 0
    modified_count = 0
    
    out_lines = []
    for line in lines:
        if not line: continue
        obj = json.loads(line)
        orig_text = obj["text"]
        new_text = apply_lexical_control(orig_text)
        
        obj["text"] = new_text
        out_lines.append(json.dumps(obj, ensure_ascii=False))
        
        processed_count += 1
        if orig_text != new_text:
            modified_count += 1
            
    out_path.write_text("\n".join(out_lines) + "\n", "utf-8")
    print(f"  Modified {modified_count}/{processed_count} records ({modified_count/max(1,processed_count)*100:.1f}%)")

def main():
    data_dir = Path("data/processed")
    
    # Process dev sets (used for polarization/FDR)
    for split in ["safe", "innov"]:
        for tag in ["dev", "train", "test"]:
            in_file = data_dir / f"d{split}_{tag}.jsonl"
            out_file = data_dir / f"d{split}_lexicalcontrol_{tag}.jsonl"
            if in_file.exists():
                process_file(in_file, out_file)
                
    print("\nLexical control corpus generation complete.")
    print("Next step: Run compute_polarization.py pointing to these new jsonl files.")
    
if __name__ == "__main__":
    main()
