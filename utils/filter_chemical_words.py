from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition, CompositionError
from tqdm import tqdm

def is_chemical_related(word):
    """
    This function checks if a word is related to chemistry.
    
    :param word: The word to be checked.
    :type word: str
    :return: A boolean indicating whether the word is chemical related or not.
    :rtype: bool
    """
    # Check if the word is an element
    try:
        el = Element(word)
        return True
    except ValueError:
        pass

    # Check if the word is a compound
    try:
        comp = Composition(word)
        return True
    except (CompositionError, ValueError, OverflowError):
        pass

    return False

def main():
    """
    This program reads a list of words from a file, filters the chemical-related words,
    and then writes these chemical-related words back to a new file.
    """
    # Read all words from the file
    input_file_path = '<Your Input File Path>'  # Replace with the actual path
    with open(input_file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f.readlines()]

    # Filter out the chemical related words
    chemical_words = [word for word in tqdm(words) if is_chemical_related(word)]

    # Save the chemical-related words to a txt file
    output_file_path = '<Your Output File Path>'  # Replace with the actual path
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for word in chemical_words:
            f.write(word + '\n')

    print(f"Chemical related words have been saved to {output_file_path}!")

if __name__ == "__main__":
    main()
