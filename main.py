import os
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Process PDFs one by one
def process_pdfs(pdf_dir, db_chroma):
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_dir, pdf_file)

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(pages)

        # Add chunks to the existing Chroma instance
        if chunks:
            db_chroma.add_documents(chunks) 

    # Persist the database after all PDFs are processed
    db_chroma.persist()

def call_rag(query, prompt_template, db_chroma):
    # Retrieve context (you might need to adjust 'k' for more/less context)
    docs_chroma = db_chroma.similarity_search_with_score(query, k=5)
    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

    # Format the prompt with context and query
    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(context=context_text, question=query)

    # Call the LLM model
    model = ChatOpenAI()
    response_text = model.predict(prompt)
    
    return response_text, context_text

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    docs_dir = "pdfs" 
    chroma_path = "chroma_db" 

    query = """
    1. What did Country Garden actually spend to build Forest City? They initially said $100B budgeted, but maybe they only spent ~$5-10B?
    2. What have they sold to, and at what price? 
    3. Forest City is only one of their projects. And what else do they have? 
    4. What are the past and present key people? What's their contact information?
    5. Who are the major competitors of Country Garden? 
    6. What has the Chinese government done to them? 
    7. Are they doing an asset sale or something like that? 
    8. How many cents on the dollar are they getting?
    """
    
    prompt_template = """
    Answer the question based only on the following context:
    {context}
    Answer the questions based on the above context: {question}.
    Provide a detailed answer.
    Don't justify your answers.
    Don't give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    """

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db_chroma = Chroma(persist_directory=chroma_path, embedding_function=embeddings) 
    
    process_pdfs(docs_dir, db_chroma)
    
    response_text, context_text = call_rag(
        query=query,
        prompt_template=prompt_template,
        db_chroma=db_chroma
    )
    print("#########################################")
    print("#########################################")
    print("#########################################")
    print(response_text) 
    print("#########################################")
    print("#########################################")
    print("#########################################")
    print(context_text)
    