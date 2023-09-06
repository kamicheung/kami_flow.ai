from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS, Chroma

from src.utils import load_embeddings, newest_csv_path

# Initialize the embeddings model globally
embeddings_model = load_embeddings()

# Vectorize the user uploaded resume into faiss (similairty search task)
def vectorize_resume():
    with open("data/raw_resume.txt","r") as f:
        resume_string = f.read()
        db = FAISS.from_texts([resume_string], embeddings_model)
        db.save_local("data/faiss_resume")

# Vectorize job descriptions into chroma (clustering and classifciation task)
def vectorize_job_descriptions():
    # Load the lastest job descriptions into a file
    path_to_job_descriptions = newest_csv_path(directory="./data")
    loader = CSVLoader(file_path=path_to_job_descriptions, encoding="utf-8")
    data = loader.load()

    # split into chunks
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    # load into Chroma
    db = Chroma.from_documents(docs, embeddings_model, persist_directory="data/chroma_db")





# Optimized Code:

# 1. Chunking instead of using a reader which is heavy on memory on large files

# def vectorize_resume():
#     with open("data/raw_resume.txt", "r") as f:
#         db = FAISS(embeddings_model)
#         chunk_size = 1024
#         while True:
#             chunk = f.readline(chunk_size)
#             if not chunk:
#                 break
#             db.add_text(chunk)
#         db.save_local("data/faiss_resume")

# 2. Use pandas to read csv into df for covenience and usability

# def vectorize_job_descriptions():
#     # load the latest job descriptions csv file
#     path_to_job_descriptions = newest_csv_path(directory="./data")
#     data = pd.read_csv(path_to_job_descriptions, encoding="utf-# 8")

#     # split into chunks
#     text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
#     docs = text_splitter.split_documents(data["description"].tolist# ())

#     # load into Chroma
#     db = Chroma.from_documents(
#         docs, embeddings_model, persist_directory="data/chroma_db"
#     )