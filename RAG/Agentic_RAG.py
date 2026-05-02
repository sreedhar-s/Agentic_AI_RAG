import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence
from typing import TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_version=os.environ.get("OPENAI_API_VERSION"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
)

embeddings = AzureOpenAIEmbeddings(
    model = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    deployment = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    api_key= os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint= os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version= os.environ.get("OPENAI_API_VERSION")
)

client = Client()

### langraph_blogs

urls = [
    "https://docs.langchain.com/oss/python/langgraph/overview",
    "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "https://docs.langchain.com/oss/python/langgraph/streaming"
]

docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=100)
doc_splits = text_splitter.split_documents(doc_list)

vectorstore_langgraph = PineconeVectorStore.from_documents(
    documents = doc_splits,
    embedding=embeddings,
    index_name = "agentic-rag-langgraph-db"
)

retriever_langgarph = vectorstore_langgraph.as_retriever()

### Retirver to retriver Tools
retriever_tool_langgraph = create_retriever_tool(
    retriever_langgarph,
    "retriever_vector_db_langgraph_blog",
    "Search and run information about langgraph"
)

### langchain blogs - Seperate vector db
langchain_urls = [
    "https://docs.langchain.com/oss/python/langchain/overview",
    "https://docs.langchain.com/oss/python/langchain/agents",
    "https://docs.langchain.com/oss/python/langchain/messages"
]

docs = [WebBaseLoader(url).load() for url in langchain_urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=100)

doc_splits = text_splitter.split_documents(docs_list)

vectorstore_langchain = PineconeVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    index_name = "agentic-rag-langchain-db"
)

retriever_langchian = vectorstore_langchain.as_retriever()

retriever_tool_langchain = create_retriever_tool(
    retriever_langchian,
    "retriever_vector_db_langchain_blog",
    "Search and run information about langchain"
)

tools = [retriever_tool_langchain, retriever_tool_langgraph]

### Langgraph Workflow
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent(state: State):
    """
    Invokes the agent model to generate a response based on the current state. Given 
    the questionm it will decide to retie using retriever tool or simply end.

    Args:
        state (messages): the Current state
    
    Returns:
        dict: the updated state with agent response appended to messages
    """

    print("---Call Agent----")
    messages = state["messages"]
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

def grade_documents(state: State) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retirved documents are relevant to the question.

    Args: 
        state (messages): The CUrrent state

    Returns: 
        str: A decision for whether the documents are relevant or not
    """

    print("---Check Relevance---")

    class grade(BaseModel):
        """ Binary score for relevance check. """
        binary_score: str = Field(description="Relevance score 'yes' or 'no' ")

    llm_with_tool = llm.with_structured_output(grade)

    prompt = PromptTemplate(
        template = """ You are a grader accessing relevance of a retrieved document to a uer qustion. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the suer question: {question} \n
        if the document contains keyword(s) or semantic meaning related to the user question, grade it a relevant. \n
        Give a binary score 'yes' orr 'no' score to indicate whether the document is relevant to the question.
        """,
        input_variables=["context", "question"]
    )

    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("--- Decision: Docs relevant")
        return "generate"
    else:
        print("--- Decision: Dcos not relevant ---")
        return "rewrite"
    
def generate(state: State):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
        dict: The updated message
    """
    print("---Generate---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content
    prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
 
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def rewrite(state: State):
    """
    Transform the query to produce a better question

    Args: 
        state (messages): The current state

    Returns: 
        dict the updated state with re-phrased question
    """

    print("---Transform Query---")
    messages = state['messages']
    question = messages[0].content

    msg = [
        HumanMessage(
            content = f"""
            Look at the input and try to reason about the underlying semantic intent / meaning. \n
            Here is the initial question:
            \n ---- \n
            {question}
            \n ---- \n
            Formlate the inroved question: 
        """
        )
    ]

    response = llm.invoke(msg)
    return {"messages": [response]}


### Workflow
builder = StateGraph(State)

builder.add_node("agent", agent)
# retrieve = ToolNode([retriever_langchian, retriever_langgarph])
builder.add_node("retrieve", ToolNode(tools))
builder.add_node("rewrite", rewrite)
builder.add_node("generate", generate)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END
    }
)

builder.add_conditional_edges(
    "retrieve",
    grade_documents
)

builder.add_edge("generate", END)
builder.add_edge("rewrite", "agent")

graph = builder.compile()   