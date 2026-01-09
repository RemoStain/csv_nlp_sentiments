from __future__ import annotations

from dataclasses import dataclass
from itertools import tee
from typing import Iterable, Optional, Union

import pandas as pd
import tkinter as tk
import tkinter.font as tkFont
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from safe_input import safe_input


# Presets


@dataclass(frozen=True)
class InputPreset:
    preset_value: int
    file_default: str
    col_default: str
    limit_default: int


INPUT_PRESETS: dict[int, InputPreset] = {
    # -1 = Manual Input Only
    -1: InputPreset(
        preset_value=-1,
        file_default="",
        col_default="",
        limit_default=0,
    ),
    # 0 = Massive Data Set
    0: InputPreset(
        preset_value=0,
        file_default="comments.csv",
        col_default="self_text",
        limit_default=5000,
    ),
    # 1 = Small Data Set
    1: InputPreset(
        preset_value=1,
        file_default="November_Comments.csv",
        col_default="Content",
        limit_default=100,
    ),
    # 2 = Filter 2
    2: InputPreset(
        preset_value=2,
        file_default="December_Comments.csv",
        col_default="Content",
        limit_default=100,
    ),

}


# UI Viewer
class SentimentViewer:
    def __init__(
        self,
        title: str = "Sentiment Viewer",
        width: int = 80,
        height: int = 25,
        console_output: bool = False,
    ):
        self.console_output = console_output
        self.root = tk.Tk()
        self.root.title(title)

        self.custom_font = tkFont.Font(family="Courier", size=24, weight="bold")

        self.text_widget = tk.Text(
            self.root,
            font=self.custom_font,
            wrap="word",
            width=width,
            height=height,
            bg="#000000",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",
        )
        self.scrollbar = tk.Scrollbar(self.root, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=self.scrollbar.set)

        self.text_widget.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.sid = SentimentIntensityAnalyzer()

        self.records: list[dict[str, object]] = []

    @staticmethod
    def value_to_hex(val: float) -> str:
        """
        Convert compound sentiment [-1,1] to hex color (red to green).
        Args:
            val (float): Compound sentiment score in range [-1, 1].
        Returns:
            str: Hex color string.
        """
        r = int(max(0, 255 * (1 - (val + 1) / 2)))
        g = int(max(0, 255 * ((val + 1) / 2)))
        b = 0
        return f"#{r:02X}{g:02X}{b:02X}"


    def add_line(self, sentence: str, score: Optional[float] = None) -> None:
        """
        Add a sentence to the text widget with color based on sentiment,
        and also store raw sentence + compound score in self.records for CSV export.
        """
        # Treat 0.0 as a valid provided score (do not fall back to re-analysis)
        if score is not None:
            compound = float(score)
            ss = {"compound": compound}
        else:
            try:
                ss = self.sid.polarity_scores(sentence)
                compound = float(ss["compound"])
            except Exception as e:
                ss = {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
                compound = 0.0
                sentence = f"{sentence} [Error analyzing sentiment: {e}]"

        # Ensure records exists
        if not hasattr(self, "records"):
            self.records = []

        # Store structured data for later CSV export (keep raw separate from display formatting)
        self.records.append(
            {
                "sentence": sentence,
                "compound": compound,
            }
        )

        colour = self.value_to_hex(compound)

        if self.console_output:
            print(sentence)
            for k in sorted(ss):
                print(f"{k}: {ss[k]}, ", end="")
            print(f"colour: {colour}\n")

        # Presentation-only formatting
        display_text = f"{sentence}  [compound: {compound}]"
        self.text_widget.insert("end", display_text + "\n")

        line_start = f"{float(self.text_widget.index('end')) - 2} linestart"
        line_end = f"{float(self.text_widget.index('end')) - 1} lineend"
        tag_name = f"line{float(self.text_widget.index('end'))}"

        self.text_widget.tag_add(tag_name, line_start, line_end)
        self.text_widget.tag_config(tag_name, foreground=colour)
        self.text_widget.see("end")

    def run(self) -> None:
        self.root.mainloop()


# CSV Loading
def load_csv_sentences(
    file_name: str = "comments.csv",
    col_name: str = "self_text",
    chunksize: int = 50000,
    limit: int = 0,
    multiline_bucket: Optional[list[str]] = None,
    treat_blankline_as_paragraph: bool = False,
) -> Optional[Iterable[str]]:
    """
    Load sentences from a CSV file column as a generator.

    Args:
        file_name (str): Path to the CSV file.
        col_name (str): Name of the column to extract sentences from.
        chunksize (int): Number of rows per chunk to read.
        limit (int): Maximum number of sentences to yield (0 = no limit).
        multiline_bucket (list[str] or None): List to store original multiline sentences.
        treat_blankline_as_paragraph (bool): If True, treat double newlines as paragraph breaks.

    Returns:
        generator or None: A generator yielding sentences, or None if no valid sentences found.
    """
    if multiline_bucket is None:
        multiline_bucket = []

    # Validation before creating the generator
    try:
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

    def generator() -> Iterable[str]:
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

            if count == 0:
                return
        except Exception:
            return

    g = generator()
    g_check, g_live = tee(g, 2)

    try:
        next(g_check)
    except StopIteration:
        return None

    return g_live


# Export
def to_csv(output_file: str, sentences: pd.DataFrame) -> None:
    """
    Save sentences to CSV.
    Expected columns: sentence, compound (or similar).
    """
    try:
        sentences.to_csv(output_file, index=False, encoding="utf-8", sep=",")
        print(f"Sentences saved to {output_file}")
    except Exception as e:
        print(f"Error saving sentences to CSV: {e}")



# App logic
def get_preset(debug_mode: int) -> InputPreset:
    preset = INPUT_PRESETS.get(debug_mode)
    if preset is None:
        raise ValueError(f"Invalid DEBUG_MODE: {debug_mode}")
    return preset


def prompt_csv_inputs(preset: InputPreset) -> tuple[str, str, int]:
    file_name = safe_input(
        str,
        f"Enter CSV file name (default: {preset.file_default}): ",
        default=preset.file_default,
    )
    col_name = safe_input(
        str,
        f"Enter column name (default: {preset.col_default}): ",
        default=preset.col_default,
    )
    limit = safe_input(
        int,
        f"Enter max number of sentences to load (0 = all, default: {preset.limit_default}): ",
        default=preset.limit_default,
    )
    return file_name, col_name, limit


def prompt_manual_sentences() -> list[str]:
    raw = safe_input(
        str,
        "No comments found. Enter sentences separated by ';': ",
        default="Hello World!;Subject is BAD BAD BAD!!;Subject is not too bad.;Subject is the best thing ever!",
    )
    return [s for s in raw.split(";") if s.strip()]


def get_sentences(preset: InputPreset) -> Union[Iterable[str], list[str]]:
    """
    Load sentences from CSV or prompt for manual input.
    Args:
        preset (InputPreset): Preset configuration.
    Returns:
        list[str] or generator: Sentences to analyze.
    """
    try:
        file_name, col_name, limit = prompt_csv_inputs(preset)
        sentences = load_csv_sentences(file_name, col_name, limit=limit)
        if sentences is None:
            return prompt_manual_sentences()

        return sentences
    except Exception as e:
        print(f"Error loading CSV sentences: {e}")
        return prompt_manual_sentences()


def compute_average_compound(sentences: Iterable[str]) -> tuple[float, int]:
    """
    Compute average compound sentiment score over sentences.
    Args:
        sentences (Iterable[str]): Sentences to analyze.
    Returns:
        tuple[float, int]: Average compound score and count of sentences.
    """
    sid = SentimentIntensityAnalyzer()
    total = 0.0
    count = 0
    for s in sentences:
        total += sid.polarity_scores(s)["compound"]
        count += 1
    avg = total / count if count > 0 else 0.0
    return avg, count


def average_line_text(avg: float, count: int) -> str:
    return f"Average compound sentiment over {count} sentences: {avg}"


def run_pipeline(
    viewer: SentimentViewer,
    sentences: Union[Iterable[str], list[str]],
    *,
    average_only: bool,
) -> None:
    """
    Run sentiment analysis pipeline.
    Args:
        viewer (SentimentViewer): Viewer to display results.
        sentences (Iterable[str] or list[str]): Sentences to analyze.
        average_only (bool): If True, compute only average sentiment.
    Returns:
        None
    """
    if average_only:
        sid = SentimentIntensityAnalyzer()
        total = 0.0
        count = 0

        for s in sentences:
            ss = sid.polarity_scores(s)
            total += ss["compound"]
            count += 1
            viewer.add_line(s)

        avg = total / count if count > 0 else 0.0
        viewer.add_line(average_line_text(avg, count), score=avg)
        print(average_line_text(avg, count))
        return


    # Need one pass that both displays lines and computes average.
    sid = SentimentIntensityAnalyzer()
    total = 0.0
    count = 0
    for s in sentences:
        ss = sid.polarity_scores(s)
        total += ss["compound"]
        count += 1
        viewer.add_line(s)

    avg = total / count if count > 0 else 0.0
    viewer.add_line(average_line_text(avg, count), score=avg)
    print(average_line_text(avg, count))
    viewer.run()
    CSV_df = pd.DataFrame(viewer.records)
    to_csv("sentiment_output.csv", CSV_df)



def main(debug_mode: int = 1) -> None:
    preset = get_preset(debug_mode)

    sentences = get_sentences(preset)

    to_console = safe_input(
        bool, "Output to console? Y/n (default: False): ", default=False
    )
    average_only = safe_input(bool, "Compute average only? Y/n (default: False): ", default=False)
    viewer = SentimentViewer(title="Comment Sentiment Log", console_output=to_console)

    try:
        run_pipeline(viewer, sentences, average_only=average_only)
    except Exception as e:
        print(f"Error processing sentences: {e}")


if __name__ == "__main__":
    DEBUG_MODE = 2  # change this integer to switch behavior
    main(DEBUG_MODE)
