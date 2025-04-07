from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from main.data_ingestion import data_ingestion
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama3-70b-8192", temperature=0.5)

chat_history = []
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    ASSESMENT_BOT_TEMPLATE = """
You are an expert assistant trained on SHL assessments and psychometric evaluations. 
You help users understand various assessments, what skills they evaluate, their target roles, 
and other relevant details extracted from titles and descriptions.

Your responses should be accurate, concise, and informative. Stick to the assessment context strictly and avoid going off-topic.

CONTEXT:
{context}

QUESTION: {input}

YOUR ANSWER:
"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ASSESMENT_BOT_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


if __name__ == "__main__":
    vstore = data_ingestion("done")
    conversational_rag_chain = generation(vstore)

    answer = conversational_rag_chain.invoke(
        {"input": "Which assessment is good for Software engineers?"},
        config={"configurable": {"session_id": "Kartish_VStest"}},
    )["answer"]
    print(answer)

    answer1 = conversational_rag_chain.invoke(
        {"input": "What was my previous question?"},
        config={"configurable": {"session_id": "Kartish_VStest"}},
    )["answer"]
    print(answer1)
