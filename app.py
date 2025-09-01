import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
import tempfile
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # temporary directory for Chroma
    persist_directory = tempfile.mkdtemp()
    
    vectorstore = Chroma.from_texts(
        texts=text_chunks, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore


def get_conversation_chain(vectorstore):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={
            "max_length": 512,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9
        }
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return conversation_chain


def handle_userinput(user_question):
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.error("Please upload and process PDFs before asking questions.")
        return

    try:
        with st.spinner("Generating response..."):
            # Adding better context to the question
            formatted_question = f"Based on the uploaded documents, {user_question}"
            response = st.session_state.conversation.invoke({"question": formatted_question})

        # Extract answer & chat history
        answer = response.get("answer", "I couldn't find a relevant answer in the documents.")
        st.session_state.chat_history = response["chat_history"]

        # Display only the latest conversation pair
        if st.session_state.chat_history:
            # Show user question
            st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
            # Show bot response
            st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
            
            # Show source documents if available
            if "source_documents" in response and response["source_documents"]:
                with st.expander("📄 Source Context"):
                    for i, doc in enumerate(response["source_documents"][:2]):  # Show top 2 sources
                        st.write(f"**Source {i+1}:**")
                        st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.write("---")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def clear_chat_history():
    """Function to clear chat history"""
    if "chat_history" in st.session_state:
        st.session_state.chat_history = None
    if "conversation" in st.session_state:
        st.session_state.conversation.memory.clear()


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon="📚",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

    st.header("Chat with multiple PDFs 📚")
    
    # Main chat interface
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Process", type="primary"):
                if pdf_docs:
                    with st.spinner("Processing PDFs..."):
                        try:
                            # Get PDF text
                            raw_text = get_pdf_text(pdf_docs)
                            
                            if not raw_text.strip():
                                st.error("No text could be extracted from the PDFs. Please check if the PDFs contain readable text.")
                                return
                            
                            st.info(f"✅ Extracted {len(raw_text)} characters")
                            
                            # Get text chunks
                            text_chunks = get_text_chunks(raw_text)
                            st.info(f"✅ Created {len(text_chunks)} text chunks")
                            
                            # Create vector store
                            with st.spinner("Creating embeddings..."):
                                vectorstore = get_vectorstore(text_chunks)
                            st.info("✅ Vector store created")
                            
                            # Create conversation chain
                            with st.spinner("Setting up conversation chain..."):
                                st.session_state.conversation = get_conversation_chain(vectorstore)
                            
                            st.session_state.processing_complete = True
                            st.success("🎉 PDFs processed successfully! You can now ask questions.")
                            
                        except Exception as e:
                            st.error(f"Error processing PDFs: {str(e)}")
                            st.error("Please try uploading different PDF files or check the error above.")
                else:
                    st.warning("Please upload at least one PDF file.")
        
        with col2:
            if st.button("Clear Chat"):
                clear_chat_history()
                st.success("Chat history cleared!")
        
        # Display processing status
        if st.session_state.processing_complete:
            st.success("✅ PDFs Ready for Chat")
            if pdf_docs:
                st.write(f"📄 {len(pdf_docs)} PDF(s) loaded")
        else:
            st.info("⏳ Upload PDFs to start")
        
        # Add some helpful tips
        with st.expander("💡 Tips for better results"):
            st.write("""
            **Good Questions:**
            - "Who are the authors of this paper?"
            - "What is the main conclusion?"
            - "Summarize the methodology"
            - "What are the key findings?"
            
            **Tips:**
            - Be specific in your questions
            - Ask about content that's actually in the documents
            - Use follow-up questions to dive deeper
            - Check the source context for verification
            """)
        
        # Model info
        with st.expander("🔧 Model Info"):
            st.write("""
            - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
            - **Vector Store**: ChromaDB
            - **LLM**: google/flan-t5-base
            - **Retrieval**: Top 4 similar chunks
            """)


if __name__ == '__main__':
    main()