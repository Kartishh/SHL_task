from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from main.data_converter import dataconverter
from dotenv import load_dotenv
load_dotenv()


GROQ_API_KEY=os.getenv("GROQ_API_KEY")
ASTRA_DB_API_ENDPOINT=os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE=os.getenv("ASTRA_DB_KEYSPACE")
HF_TOKEN = os.getenv("HF_TOKEN")

embeddings = HuggingFaceInferenceAPIEmbeddings(api_key= HF_TOKEN, model_name="BAAI/bge-base-en-v1.5")

def data_ingestion(status):

    vstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name = "shl_assessments",
        api_endpoint = ASTRA_DB_API_ENDPOINT,
        token = ASTRA_DB_APPLICATION_TOKEN,
        namespace = ASTRA_DB_KEYSPACE 
    )

    storage = status

    if storage == None:
        docs = dataconverter()
        insert_ids = vstore.add_documents(docs)
    
    else:
        return vstore
    return vstore, insert_ids


print(f"Endpoint: {ASTRA_DB_API_ENDPOINT}")
print(f"Token: {ASTRA_DB_APPLICATION_TOKEN[:6]}...")  # only partial for safety
print(f"Keyspace: {ASTRA_DB_KEYSPACE}")


if __name__ == "__main__":
    vstore, insert_ids = data_ingestion(None)
    if insert_ids:
        print(f"\n✅ Inserted {len(insert_ids)} documents.")
    else:
        print("\n⚠️ No documents inserted. Check dataconverter() or DB connection.")

    results = vstore.similarity_search("Which assessment is good for Software engineers?")
    if results:
        print(f"\n🔍 Search Results ({len(results)} results):")
        for res in results:
            print(f"\n📄 {res.page_content} [{res.metadata}]")
    else:
        print("\n❌ No similarity results found.")





