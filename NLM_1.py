from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from safe_input import safe_input
import tkinter as tk
import tkinter.font as tkFont
import pandas as pd
from itertools import islice


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

    def add_line(self, sentence):
        """Add a sentence to the text widget with color based on sentiment."""
        ss = self.sid.polarity_scores(sentence)
        colour = self.value_to_hex(ss["compound"])

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
    file_name="comments.csv", col_name="self_text", chunksize=50000, limit=0
):
    """
    Load sentences from a CSV file column as a generator.

    Args:
        file_name (str): Path to the CSV file.
        col_name (str): Name of the column containing sentences.
        chunksize (int): Number of rows to read per chunk.
        limit (int): Maximum number of sentences to yield (0 = no limit).

    Returns:
        generator or None
    """
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

    def generator():
        count = 0
        try:
            for chunk in pd.read_csv(
                file_name,
                usecols=[col_name],
                encoding="utf-8",
                encoding_errors="ignore",
                chunksize=chunksize,
            ):
                for s in chunk[col_name]:
                    s = str(s).strip()
                    if s:
                        yield s
                        count += 1
                        if limit > 0 and count >= limit:
                            return
            # If no valid sentences were found, stop generator
            if count == 0:
                return
        except Exception:
            return  # any runtime error just stops generator

    # Check if the CSV actually contains any rows
    try:
        _ = next(generator())
    except StopIteration:
        return None

    # If there is at least one row, return the generator
    return generator()


if __name__ == "__main__":
    sentences = load_csv_sentences(
        safe_input(
            str, "Enter CSV file name (default: comments.csv): ", default="comments.csv"
        ),
        safe_input(
            str, "Enter column name (default: self_text): ", default="self_text"
        ),
        limit=safe_input(
            int,
            "Enter max number of sentences to load (0 = all, default: 1000): ",
            default=1000,
        ),
    )
    if sentences is None:
        sentences = safe_input(
            str,
            "No comments found. Enter sentences separated by ';': ",
            default="Hello World!;Subject is BAD BAD BAD!!;Subject is not too bad.;Subject is the best thing ever!",
        ).split(";")

    to_console = safe_input(
        bool, "Output to console? Y/n (default: False): ", default=False
    )
    viewer = SentimentViewer(title="Comment Sentiment Log", console_output=to_console)

    for s in sentences:
        viewer.add_line(s)

    viewer.run()
