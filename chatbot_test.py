import os
import pandas as pd
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

AZURE_OPENAI_API_BASE = "https://vthackathontest1.openai.azure.com"
AZURE_OPENAI_API_KEY = "fbc23483d6de43b1a35122b379c48bd4"
AZURE_API_VERSION_CHAT = "2024-08-01-preview"  
AZURE_API_VERSION_EMBEDDINGS = "2023-05-15"    

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = AZURE_OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["OPENAI_API_VERSION"] = AZURE_API_VERSION_CHAT  
file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if file:
    all_data = pd.read_excel(file, sheet_name=None,nrows=500)  

    dataframes = []
    for sheet_name, data in all_data.items():
        if isinstance(data, pd.DataFrame):
            data.columns = data.columns.str.strip()  

            try:
                processed_data = data[['Item', 'Evidence', 'Supplier Name', 'Data_prep_date']]
                processed_data.columns = ['Item', 'Evidence', 'Supplier Name', 'Data_prep_date']  # Rename for consistency
                dataframes.append(processed_data)  # Append processed data
            except KeyError as e:
                st.warning(f"Missing columns in sheet '{sheet_name}'. Details: {e}")

    # Combine all sheets into one DataFrame
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Save the combined DataFrame to a temporary text file (tab-separated)
        txt_file_path = 'combined_output.txt'
        combined_df.to_csv(txt_file_path, sep='\t', index=False, encoding='utf-8')

        # Check if the file exists and is not empty
        if os.path.exists(txt_file_path) and os.path.getsize(txt_file_path) > 0:
            # Load the temporary text file into LangChain
            loader = TextLoader(txt_file_path, encoding="utf-8")

            # Initialize embeddings with Azure OpenAI
            embeddings = OpenAIEmbeddings(
                deployment="embedding-model-test",  # Deployment ID for embeddings model
                openai_api_key=AZURE_OPENAI_API_KEY
            )

            # Create the index with the specified embeddings
            PERSIST = False  # Set to True if you want to persist the vectorstore
            if PERSIST and os.path.exists("persist"):
                vectorstore = Chroma(persist_directory="persist", embedding_function=embeddings)
                index = VectorStoreIndexWrapper(vectorstore=vectorstore)
            else:
                # Pass embeddings explicitly to the VectorstoreIndexCreator
                index_creator = VectorstoreIndexCreator(embedding=embeddings)
                index = index_creator.from_loaders([loader])

            # Initialize the LLM (Language Model) for chat-like interaction
            llm = ChatOpenAI(
                deployment_id="gpt-4o",  # Deployment ID for the chat model
                openai_api_key=AZURE_OPENAI_API_KEY
            )

            # Initialize retrieval-based QA chain
            retriever = index.vectorstore.as_retriever(search_kwargs={"k": 3})  # Search for top 3 relevant documents
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


            st.title("Chatbot 🤖")
            st.write("Ask questions about the Excel data loaded into the system.")

            # Chat history
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            # User input
            with st.form(key="chat_form"):
                user_input = st.text_input("Your Question ", value="")
                submit_button = st.form_submit_button("Submit")

            # Process query
            if submit_button and user_input.strip():
                # Store user message
                st.session_state["messages"].append({"role": "user", "content": user_input})

                try:
                    # Generate response
                    response = qa_chain.invoke({"query": user_input})
                    bot_reply = response["result"]

                    # Store bot reply
                    st.session_state["messages"].append({"role": "bot", "content": bot_reply})
                except Exception as e:
                    bot_reply = f"An error occurred: {e}"
                    st.session_state["messages"].append({"role": "bot", "content": bot_reply})

            # Display chat history
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Bot:** {msg['content']}")
        else:
            st.error("The combined output file is either missing or empty. Please try again.")
    else:
        st.error("No valid sheets found in the uploaded Excel file.")
else:
    st.info("Please upload an Excel file to proceed.")






