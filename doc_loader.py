from langchain_community.document_loaders import CSVLoader

# Excel file ka path do
def load_csv(file_path):
# Load Excel file
    loader = CSVLoader(file_path=file_path,
    csv_args={
    'delimiter': ',',
    # 'quotechar': '"',
    # 'fieldnames': ['Index', 'Height', 'Weight']
})
    # Load and return the parsed content as a list of documents
    return loader.load()

