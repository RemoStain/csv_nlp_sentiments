from nltk.tokenize.toktok import ToktokTokenizer

from itertools import tee
from collections import Counter

import pandas as pd
from safe_input import safe_input
from dataclasses import dataclass

_tokenizer = ToktokTokenizer()


@dataclass(frozen=True)
class InputPreset:
    preset_value: int
    file_default: str
    col_default: str
    limit_default: int


INPUT_PRESETS: dict[int, InputPreset] = {
    # 0 = Gaza comments
    0: InputPreset(
        preset_value=0,
        file_default="comments.csv",
        col_default="self_text",
        limit_default=100,
    ),
    # 1 = Yasmin's comments
    1: InputPreset(
        preset_value=1,
        file_default="Yasmins_comments.csv",
        col_default="Content",
        limit_default=100,
    ),
}


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
    except Exception:
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
        except Exception:
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


def word_frequency(sentences):
    freqs = Counter()
    for s in sentences:

        freqs.update(t.lower() for t in _tokenizer.tokenize(s) if t.isalpha())
    return freqs


if __name__ == "__main__":
    DEBUG_MODE = 1  # change this integer to switch behavior

    preset = INPUT_PRESETS.get(DEBUG_MODE)
    if preset is None:
        raise ValueError(f"Invalid DEBUG_MODE: {DEBUG_MODE}")
    try:
        sentences = csv_reading_gen(
            safe_input(
                str,
                f"Enter CSV file name (default: {preset.file_default}): ",
                default=preset.file_default,
            ),
            safe_input(
                str,
                f"Enter column name (default: {preset.col_default}): ",
                default=preset.col_default,
            ),
            limit=safe_input(
                int,
                f"Enter max number of sentences to load (0 = all, default: {preset.limit_default}): ",
                default=preset.limit_default,
            ),
        )
    except Exception as e:
        print(f"Error loading CSV sentences: {e}")

        sentences = safe_input(
            str,
            "No comments found. Enter sentences separated by ';': ",
            default="Hello World!;Subject is BAD BAD BAD!!;Subject is not too bad.;Subject is the best thing ever!",
        ).split(";")

    freqs = word_frequency(sentences)

    print(freqs.most_common(10))