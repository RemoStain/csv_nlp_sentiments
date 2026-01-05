from nltk.corpus import wordnet as wn
from safe_input import safe_input


def get_synonyms(word: str) -> set[str]:
    """
    Get synonyms for a given word using WordNet.
    Args:
        word (str): Input word.
    Returns:
        set of str: Set of synonyms.
    """
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def get_antonyms(word: str) -> set[str]:
    """
    Get antonyms for a given word using WordNet.
    Args:
        word (str): Input word.
    Returns:
        set of str: Set of antonyms.
    """
    antonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                for ant in lemma.antonyms():
                    antonyms.add(ant.name())
    return antonyms 


def get_word(getting: str) -> str:
    word = safe_input(str, f"Enter a word to get {getting} : ")
    if word:
        return word.lower()
    else:
        return get_word()


def menu():
    while True:
        menu_items = [
            "Get synonyms for a word",
            "Get antonyms for a word",
        ]
        menu_lines = [f"{i}. {item}" for i, item in enumerate(menu_items, start=1)]
        menu_lines.append("0. Logout")

        for l in menu_lines:
            print(l)

        try:
            selection = "words"
            choice = safe_input(int, "Enter your choice: ")
            match choice:
                case 0:
                    print("Logging out...")
                case 1:
                    selection = "synonyms"
                    word = get_word(selection)
                    fetched = get_synonyms(word)
                case 2:
                    selection = "antonyms"
                    word = get_word(selection)
                    fetched = get_antonyms(word)
                case _:
                    print("Invalid choice.")

        except Exception as e:
            print(f"an error occurred: {e}")

        finally:
            if choice == 0:
                print("test")
                break
            if choice in [1, 2]:
                if fetched:
                    print(f"The {selection} for '{word}' are: {', '.join(fetched)}")
                else:
                    print(f"No {selection} found for '{word}'.")
            word = ""
            fetched.clear()


if __name__ == "__main__":
    menu()