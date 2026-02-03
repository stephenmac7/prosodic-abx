#!/usr/bin/env python3
"""Create MFA lexicon from IPA guide for English stress minimal pairs.

Converts IPA transcriptions from ipa_guide.txt to MFA phone format and
generates a lexicon file for Montreal Forced Aligner.
"""

import argparse
import re
from pathlib import Path

# IPA to MFA phone mapping
# Based on english_mfa.dict phone inventory
IPA_TO_MFA = {
    # Affricates (handle multi-char first)
    "d͡ʒ": "dʒ",
    "t͡ʃ": "tʃ",
    "dʒ": "dʒ",
    "tʃ": "tʃ",
    # Vowels
    "ɑː": "ɑː",
    "ɔː": "ɒː",
    "iː": "iː",
    "uː": "uː",
    "eɪ": "ej",
    "aɪ": "aj",
    "ɔɪ": "ɔj",
    "aʊ": "aw",
    "oʊ": "ow",
    "əʊ": "əw",
    "ɝ": "ɝ",
    "ɚ": "ɚ",
    "ʌ": "ɐ",
    "æ": "æ",
    "ɑ": "ɑ",
    "ɒ": "ɒ",
    "ɔ": "ɔ",
    "ə": "ə",
    "ɛ": "ɛ",
    "ɜ": "ɜ",
    "ɪ": "ɪ",
    "ʊ": "ʊ",
    "i": "i",
    "u": "u",
    "e": "e",
    "o": "o",
    "a": "a",
    # Consonants
    "ɹ": "ɹ",
    "ŋ": "ŋ",
    "θ": "θ",
    "ð": "ð",
    "ʃ": "ʃ",
    "ʒ": "ʒ",
    "ɫ": "ɫ",
    "ɡ": "ɡ",
    "p": "p",
    "b": "b",
    "t": "t",
    "d": "d",
    "k": "k",
    "g": "ɡ",
    "f": "f",
    "v": "v",
    "s": "s",
    "z": "z",
    "h": "h",
    "m": "m",
    "n": "n",
    "l": "l",
    "r": "ɹ",
    "w": "w",
    "j": "j",
}

# Target words from ipa_guide.txt
IPA_MAP = {
    "abstract": {
        "noun": "æbˌstɹækt",
        "verb": "ˌæbˈstɹækt",
    },
    "address": {
        "noun": "ˈædɹɛs",
        "verb": "əˈdɹɛs",
    },
    "conduct": {
        "noun": "ˈkɒndʌkt",
        "verb": "kənˈdʌkt",
    },
    "discharge": {
        "noun": "ˈdɪstʃɑːdʒ",
        "verb": "dɪsˈtʃɑːdʒ",
    },
    "discount": {
        "noun": "ˈdɪskaʊnt",
        "verb": "dɪˈskaʊnt",
    },
    "impact": {
        "noun": "ˈɪmpækt",
        "verb": "ɪmˈpækt",
    },
    "import": {
        "noun": "ˈɪmpɔɹt",
        "verb": "ɪmˈpɔɹt",
    },
    "increase": {
        "noun": "ˈɪnkɹiːs",
        "verb": "ɪnˈkɹiːs",
    },
    "insert": {
        "noun": "ˈɪnsɝt",
        "verb": "ɪnˈsɝt",
    },
    "insult": {
        "noun": "ˈɪnsʌlt",
        "verb": "ɪnˈsʌlt",
    },
    "permit": {
        "noun": "ˈpɝmɪt",
        "verb": "pɚˈmɪt",
    },
    "project": {
        "noun": "ˈpɹɑˌd͡ʒɛkt",
        "verb": "pɹəˈd͡ʒɛkt",
    },
    "survey": {
        "noun": "ˈsɝˌveɪ",
        "verb": "sɚˈveɪ",
    },
    "transfer": {
        "noun": "ˈtɹæns.fɚ",
        "verb": "tɹænsˈfɚ",
    },
    "transport": {
        "noun": "ˈtɹæns.pɔɹt",
        "verb": "tɹænsˈpɔɹt",
    },
    "upset": {
        "noun": "ˈʌpsɛt",
        "verb": "ˌʌpˈsɛt",
    },
}

# Carrier phrase words from MFA dictionary
MFA_CARRIER_WORDS = {
    "i": "aj",
    "say": "s ej",
    "again": "a ɡ e n",
}


def ipa_to_mfa_phones(ipa: str) -> str:
    """Convert IPA transcription to MFA phone sequence.

    Args:
        ipa: IPA string with stress markers

    Returns:
        Space-separated MFA phone sequence
    """
    # Remove stress markers and syllable boundaries
    ipa = ipa.replace("ˈ", "").replace("ˌ", "").replace(".", "")

    phones = []
    i = 0
    while i < len(ipa):
        matched = False
        # Try longest matches first (up to 3 chars for affricates with tie bar)
        for length in [3, 2, 1]:
            if i + length <= len(ipa):
                segment = ipa[i:i+length]
                if segment in IPA_TO_MFA:
                    phones.append(IPA_TO_MFA[segment])
                    i += length
                    matched = True
                    break
        if not matched:
            # Skip unknown characters (shouldn't happen with proper input)
            print(f"Warning: Unknown IPA segment '{ipa[i]}' in '{ipa}'")
            i += 1

    return " ".join(phones)


def create_lexicon(output_path: Path):
    """Create MFA lexicon file.

    Args:
        output_path: Path to write lexicon file
    """
    entries = []

    # Add carrier phrase words
    for word, phones in MFA_CARRIER_WORDS.items():
        entries.append(f"{word}\t{phones}")

    # Add target words with noun/verb variants
    for word, forms in IPA_MAP.items():
        for pos, ipa in forms.items():
            mfa_phones = ipa_to_mfa_phones(ipa)
            # Use word as key (MFA allows multiple pronunciations)
            entries.append(f"{word}\t{mfa_phones}")

    # Write lexicon
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries) + "\n")

    print(f"Created lexicon with {len(entries)} entries: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create MFA lexicon from IPA guide"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synth_data/kokoro_english_stress/lexicon.txt"),
        help="Output lexicon file path",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    create_lexicon(args.output)


if __name__ == "__main__":
    main()
