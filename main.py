import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import ttk
from functools import partial
import nltk
import re
import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('punkt')

def preprocess(text):
    """
    Preprocesses text by tokenizing, stemming, removing stopwords,
    casefolding, filtering tokens, preserving special cases, and
    handling punctuation and digits.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        list: A list of preprocessed tokens.
    """
    # Load stopwords
    stop_words = load_stopwords('Stopword-List.txt')

    # Create stemmer for reducing words to their root forms
    stemmer = PorterStemmer()

    # Define regular expressions for email and URL patterns
    email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    url_regex = r'https?://[^\s]+'

    # Define a function to handle punctuation and digits
    def clean_token(token):
        if re.match(email_regex, token) or re.match(url_regex, token):
            return token  # Preserve emails and URLs
        else:
            return re.sub(r'[^a-zA-Z0-9]', '', token)  # Remove all non-alphanumeric characters

    # Split text into words (considering special cases)
    tokens = []
    for word in re.findall(r'(?:' + email_regex + r'|' + url_regex + r'|\S+)', text):
        tokens.append(clean_token(word))

    # Preprocess tokens:
    processed_tokens = []
    for token in tokens:
        # Convert to lowercase (excluding emails and URLs)
        if not re.match(email_regex, token) and not re.match(url_regex, token):
            token = token.lower()
            token = re.sub(r'\d', '', token) # Drop digits from the token

        # Retain tokens containing at least one alphabetic character
        if any(c.isalpha() for c in token):
            if len(token) > 2:  # Exclude short tokens
                stemmed_token = stemmer.stem(token)  # Reduce word to root form
                if stemmed_token not in stop_words:  # Exclude stopwords
                    processed_tokens.append(stemmed_token)

    return processed_tokens

def build_indexes(documents, indexes_dir="indexes"):
    """
    Builds an inverted index and a positional index from a collection of documents.

    Args:
    - documents (dict): A dictionary where keys are document IDs and values are documents.
    - indexes_dir (str): The directory to store the index files.

    Returns:
    - dict: A tuple containing the inverted index and the positional index.
    """

    # Create the directory if it doesn't exist
    if not os.path.exists(indexes_dir):
        os.makedirs(indexes_dir)

    inverted_index_file = os.path.join(indexes_dir, 'invertedindex.txt')
    positional_index_file = os.path.join(indexes_dir, 'positionalindex.txt')

    if os.path.exists(inverted_index_file) and os.path.exists(positional_index_file):
        inverted_index = load_inverted_index(inverted_index_file)
        positional_index = load_positional_index(positional_index_file)
        return inverted_index, positional_index

    inverted_index = {}
    positional_index = {}

    for doc_id, document in documents.items():
        # Tokenize document on whitespace
        tokens = document.split()

        # Extract positions of terms without preprocessing
        positions = {}
        for position, token in enumerate(tokens):
            processed_token = preprocess(token)
            if not processed_token:
                continue  # Skip if token is filtered out during preprocessing
            processed_token = processed_token[0]  # Take the first token after preprocessing
            if processed_token not in positions:
                positions[processed_token] = []
            positions[processed_token].append(position)

        for token, token_positions in positions.items():
            # Update positional index
            if token not in positional_index:
                positional_index[token] = {}
            if doc_id not in positional_index[token]:
                positional_index[token][doc_id] = []  # Initialize as list
            positional_index[token][doc_id].extend(token_positions)

            # Update inverted index
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(doc_id)

    # Sort the inverted index by keys (terms) alphabetically
    inverted_index = {k: sorted(inverted_index[k]) for k in sorted(inverted_index)}

    # Sort document IDs and positions in positional index
    for term in positional_index:
        sorted_docs = sorted(positional_index[term].items(), key=lambda x: x[0])  # Sort document IDs
        positional_index[term] = {doc_id: sorted(positions) for doc_id, positions in sorted_docs}  # Sort positions

    # Sort terms alphabetically
    positional_index = {term: positional_index[term] for term in sorted(positional_index)}

    # Save indexes to files
    save_inverted_index(inverted_index, inverted_index_file)
    save_positional_index(positional_index, positional_index_file)

    return inverted_index, positional_index

def save_inverted_index(index, filename):
    with open(filename, 'w') as file:
        for term, doc_ids in index.items():
            file.write(term + ':')
            file.write(' '.join(str(doc_id) for doc_id in doc_ids))
            file.write('\n')

def save_positional_index(index, filename):
    with open(filename, 'w') as file:
        for term, postings in index.items():
            file.write(term + ':')
            for doc_id, positions in postings.items():
                file.write(f' {doc_id} [{",".join(map(str, positions))}]')
            file.write('\n')

def load_inverted_index(filename):
    index = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            if len(parts) != 2:
                # Skip this line if it doesn't contain a colon
                continue
            term, doc_ids = parts
            index[term] = [int(doc_id) for doc_id in doc_ids.split()]
    return index

def load_positional_index(filename):
    positional_index = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            if len(parts) < 2:
                # Skip this line if it doesn't contain at least one colon
                continue
            term = parts[0]
            postings_str = ':'.join(parts[1:])
            postings = {}
            postings_list = postings_str.strip().split()
            for i in range(0, len(postings_list), 2):
                if i + 1 >= len(postings_list):
                    # Skip if there's no enough elements
                    continue
                doc_id = postings_list[i]
                positions_str = postings_list[i+1][1:-1]
                if positions_str:
                    positions = list(map(int, positions_str.split(',')))
                    postings[doc_id] = positions
            positional_index[term] = postings
    return positional_index

def boolean_query(query, inverted_index, all_documents):
    """
    Performs boolean queries on an inverted index.

    Args:
    - query (str): The boolean query to be executed.
    - inverted_index (dict): The inverted index generated from documents.
    - all_documents (set): Set of all document IDs.

    Returns:
    - set: Set of document IDs that match the query.
    """
    # Tokenize the query
    query_terms = query.split()

    # Placeholder set to store the results
    result_set = set()

    # Placeholder to keep track of the last operation
    last_operation = None

    # Flag to indicate NOT operation
    not_flag = False

    # Process each term in the query
    for term in query_terms:
        if term.lower() == "and":
            last_operation = "AND"
            not_flag = False
        elif term.lower() == "or":
            last_operation = "OR"
            not_flag = False
        elif term.lower() == "not":
            # Toggle the NOT flag
            not_flag = not not_flag
        else:
            # Term is not an operator, so it's a search term
            processed_term = preprocess(term)[0]  # Preprocess the term
            term_results = set(inverted_index.get(processed_term, set()))

            print(f"Term: {term}, Term Results Type: {type(term_results)}")  # Debugging

            # Apply NOT operation if flag is set
            if not_flag:
                term_results = all_documents.difference(term_results)

            # Apply previous operation
            if last_operation == "AND":
                result_set = result_set.intersection(term_results)
            elif last_operation == "OR":
                result_set = result_set.union(term_results)
            else:
                # If there was no previous operation, initialize the result set
                result_set = term_results

    result_set = sorted(result_set)
    return result_set

def proximity_query(query, positional_index):
    """
    Performs proximity queries on a positional index.

    Args:
    - query (str): The proximity query in the format "X Y / k" where X and Y are terms and k is the proximity distance.
    - positional_index (dict): The positional index generated from documents.

    Returns:
    - set: Set of document IDs that match the proximity query.
    """
    # Tokenize the query
    query_terms = word_tokenize(query)

    # Extract the proximity distance
    proximity_distance = int(query_terms[-1])

    # preprocess the query
    query_terms = preprocess(query)

    # Placeholder set to store the results
    result_set = set()

    # Check if terms are present in the positional index
    if all(term in positional_index for term in query_terms):
        # Iterate over the positions of the first term
        for doc_id, positions in positional_index[query_terms[0]].items():
            for position in positions:
                # Check if the subsequent term is present in the same document within the proximity distance
                for i in range(1, len(query_terms)):
                    next_term_positions = positional_index[query_terms[i]].get(doc_id, [])
                    for next_position in next_term_positions:
                        if abs(next_position - position) == proximity_distance + 1:
                            result_set.add(doc_id)
                            break

    result_set = sorted(result_set)
    return result_set

def load_stopwords(filename):
    """
    Loads stopwords from a file.

    Args:
    - filename (str): The path to the file containing stopwords.

    Returns:
    - set: A set of stopwords.
    """
    stopwords = set()
    with open(filename, 'r') as file:
        for line in file:
            stopwords.add(line.strip())
    return stopwords

def load_documents(directory):
    """
    Loads documents from a directory.

    Args:
    - directory (str): The path to the directory containing documents.

    Returns:
    - dict: A dictionary where keys are document IDs and values are documents.
    """
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            doc_id = int(os.path.splitext(filename)[0])  # Extract document ID from filename
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                documents[doc_id] = file.read()
    return documents

# Define the main GUI function
def main():
    # Load documents
    documents = load_documents('ResearchPapers')
    # Build index
    inverted_index, positional_index = build_indexes(documents)
    # Create the main window
    window = tk.Tk()
    window.title("Query GUI")
    window.geometry("800x600")

    # Create a label for the query input
    query_label = tk.Label(window, text="Enter your query:")
    query_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

    # Create an entry widget for the query input
    query_entry = tk.Entry(window, width=50)
    query_entry.grid(row=0, column=1, columnspan=2, sticky="we", padx=5, pady=5)
    default_text = "Enter your query here..."
    query_entry.insert(0, default_text)
    query_entry.config(fg="grey")  # Set text color to grey
    query_entry.bind("<FocusIn>", lambda event: on_entry_click(event, query_entry, default_text))
    query_entry.bind("<FocusOut>", lambda event: on_focus_out(event, query_entry, default_text))
    query_entry.bind("<Key>", lambda event: on_key_press(event, query_entry, default_text))

    # Create a label for the query type selection
    query_type_label = tk.Label(window, text="Select query type:")
    query_type_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)

    # Create a variable to hold the selected query type
    query_type_var = tk.StringVar()
    query_type_var.set("Boolean")  # Default query type is boolean

    # Create styled buttons for selecting query type
    boolean_button = ttk.Radiobutton(window, text="Boolean", variable=query_type_var, value="Boolean")
    boolean_button.grid(row=1, column=1, sticky="w", padx=5, pady=5)
    proximity_button = ttk.Radiobutton(window, text="Proximity", variable=query_type_var, value="Proximity")
    proximity_button.grid(row=1, column=2, sticky="w", padx=5, pady=5)

    # Create a scrolled text widget for displaying the results
    results_text = scrolledtext.ScrolledText(window, width=60, height=15)
    results_text.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

    # Define a function to execute the query and display results
    def execute_query(event=None):
        query = query_entry.get()
        if query and query != default_text:
            if query_type_var.get() == "Boolean":
                result_set = boolean_query(query, inverted_index, set(documents.keys()))
            elif query_type_var.get() == "Proximity":
                # Check if the proximity query is in the correct format
                if not re.match(r'\w+ \w+ /\s*\d+', query):
                    messagebox.showwarning("Warning", "Invalid query format. Correct format is 'X Y / k'")
                    query_entry.delete(0, tk.END)  # Reset query entry
                    query_entry.insert(0, default_text)
                    query_entry.config(fg="grey")
                    query_entry.icursor(0)
                    return
                result_set = proximity_query(query, positional_index)
            else:
                messagebox.showwarning("Warning", "Invalid query type selected.")
                return
            results_text.insert(tk.END, f"Query: {query}\n")
            if result_set:
                results_text.insert(tk.END, "Matching Documents:\n")
                for doc_id in result_set:
                    results_text.insert(tk.END, f"Document ID: {doc_id}\n")
                results_text.insert(tk.END, "-" * 50 + "\n")  # Add a separator line
            else:
                results_text.insert(tk.END, "No matching documents found.\n")
                results_text.insert(tk.END, "-" * 50 + "\n")  # Add a separator line

            # Reset query entry
            query_entry.delete(0, tk.END)
            query_entry.insert(0, default_text)
            query_entry.config(fg="grey")
            query_entry.icursor(0)

        else:
            messagebox.showwarning("Warning", "Please enter a query.")

    # Bind the Enter key to execute_query function
    window.bind("<Return>", execute_query)

    # Create a button to execute the query
    query_button = ttk.Button(window, text="Execute Query", command=execute_query)
    query_button.grid(row=3, column=0, columnspan=3, sticky="we", padx=5, pady=5)

    # Create a button to clear the output
    def clear_output():
        results_text.delete('1.0', tk.END)
    clear_button = ttk.Button(window, text="Clear Output", command=clear_output)
    clear_button.grid(row=4, column=0, columnspan=3, sticky="we", padx=5, pady=5)

    # Configure row and column weights for resizing
    window.grid_rowconfigure(2, weight=1)
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=1)
    window.grid_columnconfigure(2, weight=1)

    # Function to handle entry widget click event
    def on_entry_click(event, entry, default_text):
        """Function to handle entry widget click event."""
        if entry.get() == default_text:
            entry.delete(0, tk.END)
            entry.config(fg="black")

    # Function to handle focus out event
    def on_focus_out(event, entry, default_text):
        """Function to handle focus out event."""
        if not entry.get():
            entry.insert(0, default_text)
            entry.config(fg="grey")

    # Function to handle key press event
    def on_key_press(event):
        """Function to handle key press event."""
        if query_entry.get() == default_text:
            query_entry.delete(0, tk.END)
            query_entry.config(fg="black")  # Set text color to black

    # Bind event for handling typing
    query_entry.bind("<Key>", on_key_press)
    # Run the main event loop
    window.mainloop()

if __name__ == "__main__":
    main()