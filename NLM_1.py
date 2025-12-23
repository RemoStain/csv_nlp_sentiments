from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from safe_input import safe_input

import tkinter as tk
import tkinter.font as tkFont
import pandas as pd
from itertools import tee

from dataclasses import dataclass


@dataclass(frozen=True)
class InputPreset:
    preset_value: int
    file_default: str
    col_default: str
    limit_default: int


INPUT_PRESETS: dict[int, InputPreset] = {
    # 0 = Gaza comments
    0: InputPreset(
        preset_value = 0,
        file_default="comments.csv",
        col_default="self_text",
        limit_default=100,
    ),
    # 1 = Yasmin's comments
    1: InputPreset(
        preset_value = 1,
        file_default="Yasmins_comments.csv",
        col_default="Content",
        limit_default=100,
    ),
}


class SentimentViewer:
    def __init__(
        self,
        title="Sentiment Viewer",
        width=80,
        height=25,
        console_output: bool = False,
    ):
        self.console_output = console_output
        self.root = tk.Tk()
        self.root.title(title)
        self.custom_font = tkFont.Font(family="Courier", size=24, weight="bold")
        # Text widget with scrollbar
        self.text_widget = tk.Text(
            self.root,
            font=self.custom_font,
            wrap="word",
            width=width,
            height=height,
            bg="#000000",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",  # Cursor
        )
        self.scrollbar = tk.Scrollbar(self.root, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=self.scrollbar.set)

        self.text_widget.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Sentiment analyzer
        self.sid = SentimentIntensityAnalyzer()

    @staticmethod
    def value_to_hex(val):
        """Convert compound sentiment [-1,1] to hex color (red to green)."""
        r = int(max(0, 255 * (1 - (val + 1) / 2)))
        g = int(max(0, 255 * ((val + 1) / 2)))
        b = 0
        return f"#{r:02X}{g:02X}{b:02X}"

    def add_line(self, sentence, score:float=None):
        """Add a sentence to the text widget with color based on sentiment."""
        if score:
            colour = self.value_to_hex(score)
            ss = {"compound": score}


        else:
            try:
                ss = self.sid.polarity_scores(sentence)
                colour = self.value_to_hex(ss["compound"])
            except Exception as e:
                ss = {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
                colour = "#FFC0CB"
                sentence += f" [Error analyzing sentiment: {e}]"

        # ! Console output !
        if self.console_output:
            print(sentence)
            for k in sorted(ss):
                print(f"{k}: {ss[k]}, ", end="")
            print(f"colour: {colour}\n")

        # Add the score to the sentence
        sentence = f"{sentence}  [compound: {ss['compound']}]"

        # Insert the line
        self.text_widget.insert("end", sentence + "\n")

        # Tag the line with color
        line_start = f"{float(self.text_widget.index('end')) - 2} linestart"
        line_end = f"{float(self.text_widget.index('end')) - 1} lineend"
        tag_name = f"line{float(self.text_widget.index('end'))}"
        self.text_widget.tag_add(tag_name, line_start, line_end)
        self.text_widget.tag_config(tag_name, foreground=colour)

        # Auto-scroll
        self.text_widget.see("end")

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()


def load_csv_sentences(
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


if __name__ == "__main__":
    DEBUG_MODE = 1  # change this integer to switch behavior

    preset = INPUT_PRESETS.get(DEBUG_MODE)
    if preset is None:
        raise ValueError(f"Invalid DEBUG_MODE: {DEBUG_MODE}")
    try:
        sentences = load_csv_sentences(
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

    to_console = safe_input(
        bool, "Output to console? Y/n (default: False): ", default=False
    )
    viewer = SentimentViewer(title="Comment Sentiment Log", console_output=to_console)
    if preset.preset_value >= 0:
        # extra logic to get averages of sentiments without printing lines
        try:
            sid = SentimentIntensityAnalyzer()
            total_compound = 0.0
            count = 0
            for s in sentences:
                ss = sid.polarity_scores(s)
                total_compound += ss["compound"]
                count += 1
            average_compound = total_compound / count if count > 0 else 0.0
            print(f"Average compound sentiment over {count} sentences: {average_compound}")
            viewer.add_line(f"Average compound sentiment over {count} sentences: {average_compound}", score=average_compound)
        except Exception as e:
            print(f"Error processing sentences for averages: {e}")

    else:
        try:
            for s in sentences:
                viewer.add_line(s) 

            viewer.run()
        except Exception as e:
            print(f"Error processing sentences: {e}")
    
