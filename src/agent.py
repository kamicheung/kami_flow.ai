import logging

import coloredlogs
import langchain

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import SystemMessagePromptTemplate

from src.job_search import selenium_scrape, scrape_google_jobs
from src.utils import is_file_old_or_nonexistant, load_embeddings, load_llm
from src.vector_db import vectorize_resume, vectorize_job_descriptions
from src.tools import load_tools

from langchain.vectorstores import FAISS
from langchain.schema import Document

langchain.debug = True

# Initialize the logger to self
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

# Initialize embeddings and llm from utils
embeddings = load_embeddings()
llm = load_llm()


def load_agent(desired_job_title, mb_type, user_interests):
    # Convert desired job title to lower case and underscored
    job_title = desired_job_title.replace(" ", "_").lower()
    # Get job descriptions
    file_path = f"data/scraped_job_descriptions_{job_title}.csv"
    if is_file_old_or_nonexistant(file_path):
        logger.info(
            f"Scraped job descriptions was created more than 3 days ago or does not exist. \
            Scraping Google Jobs for fresh {desired_job_title} job descriptions."
        )
        selenium_scrape()
    else:
        logger.info('Skipping scrapping Google jobs')
    
    logger.info('Vectorizing user uploaded resume')
    vectorize_resume()

    if is_file_old_or_nonexistant(file_path):
        logger.info(f"Vectorizing scraped job descriptions for {desired_job_title} ")
        vectorize_job_descriptions()
        logger.info(f"Vectorized scraped job descriptions for {desired_job_title}")
    else:
        logger.info(f"Skipping vectorizing scraped job descriptions for {desired_job_title}")


    """Use retrieval to select the set of tools to use to answer an agent query.
    This is useful as the agent has trouble selecting the relevant tools.
    https://python.langchain.com/docs/modules/agents/how_to/custom_agent_with_tool_retrieval#tool-retriever
    """

    # Initialize locally for faster access to load tools
    all_tools = load_tools()
    
    # creates a vector store and retriever for a list of tools. 
    # The vector store and retriever are used to retrieve similar 
    # tools based on a user's query, allowing for efficient and accurate tool recommendations.
    docs = [
        Document(page_content=t.description, metadata={"index": i})
        for i, t in enumerate(all_tools)
    ]
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()

    # retrieves relevant tools based on a user's query. By using a vector store and retriever 
    # to identify the most relevant tools, the function is able to provide accurate and efficient 
    # tool recommendations to the user.
    def get_tools(query="Summarize my resume"):
        docs = retriever.get_relevant_documents(query)
        return [all_tools[d.metadata["index"]] for d in docs]

    tools = get_tools()
    tool_names = [tool.name for tool in tools]

    # Initialize Redis database for chat history
    message_history = RedisChatMessageHistory(
        session_id='flow_agent', url="redis://localhost:6379/0", ttl=600
    )

    # Initialize memory with Redis
    memory = ConversationBufferMemory(
        memory_key='chat history', chat_memory=message_history, return_messages=True
    )

    chat_agent = initialize_agent(
        agent=AgentType.OPENAI_MULTI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
        allowed_tools=tool_names,
        agent_kwargs={"system_message": system_message_prompt},
    )
    logger.info("Chat Agent loaded!")

    return chat_agent





