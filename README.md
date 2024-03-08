# Boolean-Model-for-IR
Implementation of a Boolean Model for Information Retrieval. Preprocessing, indexing, executing boolean and prioximity queries, and GUI for interaction. Sample data included.

This repository contains a Python script for a simple GUI application that allows users to execute boolean and proximity queries on a collection of documents. The application provides an interface for entering queries, selecting query types, executing queries, and viewing the results.

Features
Boolean Queries: Users can enter boolean queries to retrieve documents based on boolean operators such as AND, OR, and NOT.
Proximity Queries: Users can perform proximity queries to find documents where specific terms occur within a certain distance of each other.
Preprocessing: The application preprocesses the input text, including tokenization, stemming, stopword removal, and filtering, to enhance query accuracy.
Indexing: The script builds inverted and positional indexes from a collection of documents for efficient querying.
Graphical User Interface (GUI): The application provides a user-friendly GUI using Tkinter, allowing users to interact with the system easily.
Requirements
Python 3.x
NLTK (Natural Language Toolkit)
Tkinter (for GUI)
How to Use
Clone the Repository: Clone this repository to your local machine using the following command:

bash
Copy code
git clone https://github.com/your-username/document-query-gui.git
Install Dependencies: Install the required Python dependencies using pip:

Copy code
pip install nltk
Download NLTK Data: Run the following Python code snippet to download the necessary NLTK data:

python
Copy code
import nltk
nltk.download('punkt')
Run the Application: Navigate to the cloned directory and run the Python script document_query_gui.py:

bash
Copy code
cd document-query-gui
python document_query_gui.py
Interact with the GUI: Once the application is running, you can interact with the GUI to enter queries, select query types, execute queries, and view the results.

Sample Data
The application comes with a sample collection of documents stored in the ResearchPapers directory. These documents are used to build indexes and demonstrate query functionality.

File Descriptions
document_query_gui.py: The main Python script containing the implementation of the GUI application.
Stopword-List.txt: A text file containing a list of stopwords used for preprocessing.
ResearchPapers/: A directory containing sample research papers (text files) used for indexing and querying.
Output
Upon executing a query, the application displays the matching documents along with their document IDs.

Author
This GUI application was developed by [Your Name]. Feel free to reach out with any questions or feedback.
