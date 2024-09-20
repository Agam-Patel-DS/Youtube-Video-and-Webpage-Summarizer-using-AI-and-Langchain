import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import WebBaseLoader


# Streamlit App
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website", page_icon="A")
st.title("Langchain: Summarize text from Yt and Web")
st.subheader("Summarize URL - by Agam Patel")

# Get the Groq API key and URL input field
with st.sidebar:
  groq_api_key=st.text_input("Groq API Key", value="", type="password")

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len
)

chunks_prompt="""
Write a concise and short summary of the following speech:
Speech:{text}
Summary:
"""

map_prompt_template=PromptTemplate(
    input_variables=["text"],
    template=chunks_prompt
)

final_prompt="""
Provide hte final summary with these important points.
Add a motivational title and start a precise summary with an introduction nad provide the summary in number points.
Speech:{text}
"""

final_prompt_template=PromptTemplate(
    input_variables=["text"],
    template=final_prompt
)

generic_url=st.text_input("URL", label_visibility="collapsed")

#button
if st.button("Summarize the content!"):
  #valilidate the input
  if not groq_api_key.strip() or not generic_url.strip():
    st.error("Please provde the information!")
  elif not validators.url(generic_url):
    st.error("Please enter valid url. Only YouTube video and Web url are accepted!")

  else:
    try:
      with st.spinner("Waiting..."):
        ##loading the website or yt video data
        if "youtube.com" in generic_url: #for youtube link
          loader=YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
        else: #for webpage link
          loader = WebBaseLoader(generic_url)
        docs=loader.load()
        docs=text_splitter.split_documents(docs)

        llm=ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

        ## Chain for summarization
        chain=load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=final_prompt_template, verbose=False)
        output_summary=chain.run(docs)
        st.success(output_summary)
    except Exception as e:
      st.exception(e)




