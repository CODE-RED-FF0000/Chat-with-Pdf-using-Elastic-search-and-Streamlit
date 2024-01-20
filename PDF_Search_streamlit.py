import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from indexmapping import indexMapping
import streamlit as st

st.title('Semantic Search')
pdf = st.file_uploader("Upload your PDF", type='pdf')


if pdf is not None:
    doc_reader = PdfReader(pdf)
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    df = pd.DataFrame(texts, columns=['chunks'], dtype=str)
    #Converting the chunks into vectors
    model = SentenceTransformer('all-mpnet-base-v2')
    df['chunksvector'] = df['chunks'].apply(lambda x: model.encode(x))

    pdfindex = pdf.name
    # Connecing to Elasticsearch
    es_connection = Elasticsearch(
        "https://localhost:9200", basic_auth=("elastic", "1n9Z=V2cRkd8K==dC5C7"),
        ca_certs=("/home/sasi/PycharmProjects/pythonProject/elasticsearch-8.11.3-linux-x86_64/elasticsearch-8.11.3/config/certs/http_ca.crt"),
        timeout=3000
        )

    record_list = df.to_dict("records")
    for vectors in record_list:
        es_connection.index(index="chunk",document=vectors)

input_keyword = st.text_input('Ask your question about the pdf')
def search(input_keyword):
    vector_of_input_keyword = model.encode(input_keyword)
    query = {
        "field": "chunksvector",
        "query_vector": vector_of_input_keyword,
        "k": 3
    }

    res = es_connection.knn_search(index="chunk", knn=query, source=['chunks'])
    results = (res["hits"]["hits"])
    return results

if st.button('Search'):
    if input_keyword:
        results = search(input_keyword)
        st.subheader('Search Results')
        for result in results:
            with st.container():
                if '_source' in result:
                    st.write(f"{result['_source']['chunks']}")
                    st.divider()



