import json
import re
import pandas as pd

from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.tools import DuckDuckGoSearchRun, tool
from langchain.vectorstores import FAISS, Chroma

from src.utils import load_embeddings, load_llm, read_newest_csv

from sklearn.metrics.pairwise import cosine_similarity

# K(X, Y) = <X, Y> / (||X||*||Y||)

# where X and Y are the two vectors being compared, <X, Y> is the dot product of X and Y, 
# and ||X|| and ||Y|| are the L2 norms of X and Y, respectively.
# computes similarity as the normalized dot product of X and Y, to scale for range -1 to 1 and to account for angle and magnitude difference (high variance)
# L2 is used because it is more sensistive to differences in the magnitude of the vector's components

# Pairwise computations are commonly used in machine learning and data analysis tasks, such as 
# clustering, classification, and dimensionality reduction. By computing the pairwise distances 
# or similarities between a set of vectors, it is possible to identify patterns and relationships 
# in the data that might not be apparent from a simple inspection of the vectors themselves.


# Initialize the llm globally
llm = load_llm()

# Initialize the embeddings model globally
embeddings_model = load_embeddings()


# Load resume and user info data and turn into one string each
resume_path = "data\raw_resume.txt"
with open(resume_path, "r") as file:
    resume = file.read().replace("\n","")
user_info = "data\user_info.txt"
with open(user_info, "r") as file:
    user_info = file.read().replace("\n","")

# Embed the user persona
user_persona = resume + "\n\n" + user_info
resume_embedding = embeddings_model.embed_documents([user_persona])

def extract_salary_range(data):
    salary_pattern = r"\$\d+\.?\d*K–\$\d+\.?\d*K a month"
    match = re.findall(salary_pattern, data)
    return match[0] if match else None

@tool
def job_recommendation():
    """useful for when you need to provide job recommendations to the user, leveraging both their uploaded resume and a dataset of scraped job postings. \
    Output is a JSON in the following format: '[{"title":"<title>","company":"<company>","salary_range":"<salary_range>","link":"<link>","similarity":"<similarity>"},...]' """

    # Read latest csv and load job descriptions
    df_jobs = read_newest_csv(directory="./data")

    # Embed job descriptions in batch
    job_descriptions = df_jobs["description"].tolist()
    embedded_descriptions = embeddings_model.embed_documents(job_descriptions)

    # Perform cosine-similarity
    similarities = cosine_similarity(resume_embedding, embedded_descriptions)
    df_jobs['similarity'] = similarities.flatten() # flatten to 1d array and keeps order by default from df 

    # sort, extract salary and drop uneeded columns from job descriptions
    df_jobs_sorted = df_jobs.sort_values(by='similarity', ascending=False)
    df_jobs_sorted['salary'] = df_jobs_sorted['extensions'].apply(extract_salary_range)
    df_jobs_sorted.drop(columns=["description", "location", "via", "extensions"], inplace=True)

    # Output top 10 results to json, we use json for inter-system usability
    output = df_jobs_sorted.head(10).to_dict('records')
    output_json = json.dumps(output)

    return output_json


def load_tools():
    # load resume vectors and use RetrievalQA model on it
    resume_db = FAISS.load_local("data\faiss_resume", embeddings_model)
    resume_retriever = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff', # Chain that combines documents by stuffing into context.
        retriever=resume_db.as_retriever()
    )

    # load job descriptions and use RetrievalQA model on it
    job_description_db = Chroma(
        embedding_function=embeddings_model, persist_directory="data\chroma_db"
    )
    job_description_retriever = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff', retriever=job_description_db.as_retriever()
    )

    search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="Scraped-Job-Description-QA-System",
            func=job_description_retriever.run,
            description="useful for when you need to answer questions about job descriptions and postings that are based on the user's desired job title. Input should be a fully formed question.",
        ),
        Tool(
            name="Resume-QA-System",
            func=resume_retriever.run,
            description="useful for when you need to answer questions that the user asks about their resume. Use this if the question is about the user's resume, like 'what are the transferrable skills from my resume?', 'what are the top skills from my resume?' or 'summarize my resume'. Input should be a fully formed question.",
        ),
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current career landscape, courses and trends with an internet search. You should ask targeted questions.",
        ),
        job_recommendation,
    ]

    return tools