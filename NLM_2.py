import json
from pathlib import Path
from collections import Counter
from typing import Iterable, Iterator

import pandas as pd
from nltk.tokenize.toktok import ToktokTokenizer

from safe_input import safe_input
from error_handling_CW import error_handling
from dataclasses import dataclass
from itertools import tee


# Configuration
STOPWORD_FILE = Path("stopwords.json")
CHUNKSIZE = 50000


@dataclass(frozen=True)
class InputPreset:
    file_default: str
    col_default: str
    limit_default: int


INPUT_PRESETS: dict[int, InputPreset] = {
    0: InputPreset(
        file_default="comments.csv", col_default="self_text", limit_default=1000
    ),
    1: InputPreset(
        file_default="Yasmins_comments.csv", col_default="Content", limit_default=100
    ),
    2: InputPreset(
        file_default="inventory.txt", col_default="Row1", limit_default=300
    ),
}


# Tokenizer
_tokenizer = ToktokTokenizer()


# CSV sentence stream
def csv_reading_gen(
    file_name="comments.csv",
    col_name="self_text",
    chunksize=50000,
    limit=0,
    multiline_bucket=None,
    treat_blankline_as_paragraph=False,
):
    """
    Load sentences from a CSV file column as a generator.

    Args:
        file_name (str): Path to the CSV file.
        col_name (str): Name of the column containing sentences.
        chunksize (int): Number of rows to read per chunk.
        limit (int): Maximum number of sentences to yield (0 = no limit).
        multiline_bucket (list): List to store multi-line entries.
        treat_blankline_as_paragraph (bool): If True, only treat double newlines as multi-paragraph.

    Returns:
        generator or None: A generator yielding sentences, or None if no valid sentences found.
        list: List of multi-line entries.
    """

    if multiline_bucket is None:
        multiline_bucket = []

    # Try reading the CSV first (catch any exception before creating generator)
    try:
        # Quick check if file exists and column is valid
        _ = next(
            pd.read_csv(
                file_name,
                usecols=[col_name],
                encoding="utf-8",
                encoding_errors="ignore",
                chunksize=chunksize,
            )
        )
    except Exception as e:
        error_handling(e)
        return None

    def is_multiparagraph(text: str) -> bool:
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        return ("\n\n" in t) if treat_blankline_as_paragraph else ("\n" in t)

    def generator():
        count = 0
        row_offset = 0
        try:
            for chunk in pd.read_csv(
                file_name,
                usecols=[col_name],
                encoding="utf-8",
                encoding_errors="ignore",
                chunksize=chunksize,
            ):
                # dropna() preserves the chunk's index; we convert to global row numbers using row_offset
                series = chunk[col_name].dropna()

                for local_i, s in series.items():
                    s = str(s).strip()
                    if not s:
                        continue

                    global_row = row_offset + int(local_i)
                    prefixed = f"{global_row} {s}"

                    if is_multiparagraph(s):
                        prefixed_fixed = prefixed.replace("\n", " ").replace("\r", " ")
                        multiline_bucket.append(prefixed)

                        yield prefixed_fixed
                        continue

                    yield prefixed
                    count += 1

                    if limit > 0 and count >= limit:
                        return

                row_offset += len(chunk)

            # If no valid sentences were found, stop generator
            if count == 0:
                return
        except Exception as e:
            error_handling(e)
            return  # any runtime error just stops generator

    # Create generator
    g = generator()

    # Duplicate it so we can safely peek
    g_check, g_live = tee(g, 2)

    # Check if at least one item exists
    try:
        next(g_check)
    except StopIteration:
        return None

    # Return the untouched stream
    return g_live


# Stopword loading (isolated)
def load_stopwords(sentences: Iterable[str]) -> set[str]:
    """
    Load stopwords from disk or build them if not present.
    Args:
        sentences (iterable of str): Input sentences for building stopwords if needed.
    Returns:
        set of str: Set of stopwords.
    """

    if not STOPWORD_FILE.exists():

        from build_stopwords import build_stopwords

        build_stopwords(sentences, output=str(STOPWORD_FILE))
    else:
        pass

    with STOPWORD_FILE.open("r", encoding="utf-8") as f:
        return set(json.load(f))


# Word frequency
def word_frequency(
    sentences: Iterable[str],
    stopwords: set[str],
) -> Counter[str]:
    """
    Compute word frequency from sentences, excluding stopwords.
    Args:
        sentences (iterable of str): Input sentences.
        stopwords (set of str): Set of stopwords to exclude.
    Returns:
        Counter[str]: Word frequency counter.
    """
    freqs: Counter[str] = Counter()

    for s in sentences:
        for t in _tokenizer.tokenize(s):
            w = t.lower()
            if w.isalpha() and w not in stopwords:
                freqs[w] += 1

    return freqs


# Main
def main(DEBUG_MODE) -> None:

    preset = INPUT_PRESETS.get(DEBUG_MODE)
    if preset is None:
        raise ValueError(f"Invalid DEBUG_MODE: {DEBUG_MODE}")
    try:

        file_name = safe_input(
            str,
            f"Enter CSV file name (default: {preset.file_default}): ",
            default=preset.file_default,
        )
        column_name = safe_input(
            str,
            f"Enter column name (default: {preset.col_default}): ",
            default=preset.col_default,
        )
        limit = safe_input(
            int,
            f"Enter max number of sentences to load (0 = all, default: {preset.limit_default}): ",
            default=preset.limit_default,
        )

    except Exception as e:
        print(f"Error loading CSV sentences: {e}")
        sentences = safe_input(
            str,
            "No comments found. Enter sentences separated by ';': ",
            default="Hello World!;Subject is BAD BAD BAD!!;Subject is not too bad.;Subject is the best thing ever!",
        ).split(";")
    try:
        if file_name.endswith(".txt"):
            df = pd.read_csv(
                file_name,
                header=None,
                names=["Row1"],
            )
            df.dropna()
            df.to_csv("text_to_csv.csv", index=False)

            file_name = "text_to_csv.csv"
            column_name = "Row1"
            limit = 0
            # raise ValueError("Text file input is not supported in this version.")

        else:
            pass
    except Exception as e:
        print(f"Error processing text file: {e}")
        return

    # csv_reading_gen is one-pass, so we create it twice if stopwords need building
    sentences_for_stopwords = csv_reading_gen(
        file_name,
        column_name,
        limit=limit,
    )

    stopwords = load_stopwords(sentences_for_stopwords)

    # Re-create the stream for actual processing
    sentences = csv_reading_gen(
        file_name,
        column_name,
        limit=limit,
    )

    freqs = word_frequency(sentences, stopwords)

    for word, count in freqs.most_common(20):
        print(f"{word}: {count}")


if __name__ == "__main__":
    DEBUG_MODE = 2  # change this integer to switch behavior

    main(DEBUG_MODE)
