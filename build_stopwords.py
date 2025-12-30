# language: python
import json
from collections import Counter
from typing import Iterable
from nltk.tokenize.toktok import ToktokTokenizer

_tokenizer = ToktokTokenizer()

# Fixed policy (no external "coverage" parameter):
# - primary: words appearing in >= max(ceil(0.90 * N), N - 25) documents
# - fallback: top K by document frequency if primary is empty
# - K = 15
FALLBACK_TOP_K = 100
TOLERATE_MISSES = 25
FRACTION = 0.90


def build_stopwords(
    sentences: Iterable[str],
    *,
    output: str = "stopwords.json",
) -> None:
    """
    Build a stopword list from the provided sentences and save to a JSON file.

    Args:
        sentences (Iterable[str]): An iterable of sentences (documents).
        output (str): Path to the output JSON file for stopwords.
    """
    df = Counter()
    total_docs = 0 # number of documents processed

    for s in sentences:
        total_docs += 1
        seen = {
            t.lower()
            for t in _tokenizer.tokenize(s)
            if t.isascii()
        } # unique alphabetic tokens in this document
        if seen:
            df.update(seen) # update document frequency counts

    if total_docs == 0:
        raise ValueError("No sentences provided to build_stopwords")

    if not df:
        # No alphabetic tokens at all
        stopwords = []
    else:
        frac_threshold = int((FRACTION * total_docs) + 0.999999) # ceil without math.ceil
        miss_threshold = max(total_docs - TOLERATE_MISSES, 1) # total docs - misses, at least 1
        threshold = max(frac_threshold, miss_threshold) # use the more stringent one

        stopwords = sorted([w for w, c in df.items() if c >= threshold]) # primary criterion

        if not stopwords:
            # Deterministic fallback: take top-K most ubiquitous words
            stopwords = [w for w, _ in df.most_common(FALLBACK_TOP_K)]

    with open(output, "w", encoding="utf-8") as f:
        json.dump(stopwords, f, indent=2)


    print(f"Stopwords saved to {output}, total stopwords: {len(stopwords)}")

if __name__ == "__main__":
    # Example usage
    example_sentences = [
        "This is a sample sentence.",
        "This sentence is another example.",
        "Sample sentences are useful for testing."
    ]
    build_stopwords(example_sentences, output="stopwords.json")