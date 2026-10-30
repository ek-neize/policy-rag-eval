"""
Data-quality check: find invisible or unusual characters in the documents.

Some characters take up no space on screen (for example a "soft hyphen" or a
"zero-width space"), so two words can LOOK glued together while really being
separated by something invisible. This script counts them and shows examples.

Usage:
  python src/check_text.py
  python src/check_text.py "avoid prepayment"     # also show what follows a phrase
"""

import sys
import unicodedata
from collections import Counter
from pathlib import Path

DOCS = Path("data/raw/docs")


def describe(ch):
    return f"U+{ord(ch):04X} {unicodedata.name(ch, 'UNKNOWN')}"


def is_unusual(ch):
    # Cf = invisible "format" characters; Zs = space characters other than a normal space.
    return (unicodedata.category(ch) in ("Cf", "Zs") and ch != " ") or ch == "\t"


def main():
    counts, examples = Counter(), {}
    for path in sorted(DOCS.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, ch in enumerate(text):
            if is_unusual(ch):
                counts[ch] += 1
                if ch not in examples:
                    snippet = text[max(0, i - 25):i + 25].replace("\n", " ")
                    examples[ch] = (path.name, snippet.replace(ch, "[*]"))

    if not counts:
        print("No unusual characters found.")
    for ch, n in counts.most_common():
        file, snippet = examples[ch]
        print(f"{describe(ch):45s} found {n} times")
        print(f"   e.g. {file}: ...{snippet}...   ([*] marks the character)\n")

    if len(sys.argv) > 1:
        phrase = sys.argv[1]
        print(f'Characters right after "{phrase}":')
        for path in sorted(DOCS.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            start = text.find(phrase)
            while start != -1:
                following = text[start + len(phrase):start + len(phrase) + 6]
                print(f"   {path.name}: {[describe(c) if c != ' ' else 'space' for c in following]}")
                start = text.find(phrase, start + 1)


if __name__ == "__main__":
    main()
