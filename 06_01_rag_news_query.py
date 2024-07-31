import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

#####################################################################################
# RAG 프로세스
# 	https://www.youtube.com/watch?v=1scMJH93v0M
# 1. 문서 로딩 (Document Loading) : 다양한 형식의 문서에서 텍스트 추출
# 2. 문서 분할 (Splitting) : 긴 문서를 더 작은 chunk로 분할
# 3. 임베딩 생성 (Enbedding) : chunk 를 벡터로 변환
# 4. 벡터 저장소 구축 (Vector Database) : 임베딩 결과인 벡터를 벡터 데이터베이스에 저장
# --> 사용자 질문 설정 -->
# 5. 쿼리 처리 (Query-Retriever) : 벡터 DB 에서 참고할 문서 검색
# 6. 검색된 문서를 첨부해서 PROMPT 생성
# 7. LLM에 질문
#####################################################################################

# 1. 문서 로딩 (Document Loading)
loader = WebBaseLoader(
    web_paths=("https://www.bbc.com/korean/articles/cl4yml4l6j1o",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["bbc-1cvxiy9", "bbc-fa0wmp"]},
        )
    ),
)
docs = loader.load()
print(f"문서의 수: {len(docs)}")

# 2. 문서 분할 (Splitting)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(f"split size: {len(splits)}")

# 3. 임베딩 생성 (Enbedding)
embeddings = OllamaEmbeddings(model="llama3.1")

# 4. 벡터 저장소 구축 (Vector Database)
vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
# 4-1. 쿼리 저장소 검색을 위한 retriever 생성
retriever = vector_store.as_retriever()

# PROMPT Template 생성
prompt = PromptTemplate.from_template(
"""당신은 질문-답변(Question-Answering)을 수행하는 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
질문과 관련성이 높은 내용만 답변하고 추측된 내용을 생성하지 마세요. 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

# Ollama 초기화
llm = Ollama(
    model="llama3.1",
    temperature=0
)

# 체인을 생성합니다.
chain = prompt | llm | StrOutputParser()

# 테스트 할 질문
question = "극한 호우의 원인은 무엇인가?"

# 5. 쿼리 처리 (Query-Retriever) : 벡터 DB 에서 참고할 문서 검색
retrieved_docs = retriever.invoke(question)
print(f"retrieved size: {len(retrieved_docs)}")
combined_docs = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 6. 검색된 문서를 첨부해서 PROMPT 생성
formatted_prompt = {"context": combined_docs, "question": question}

# 7. LLM에 질문
for chunk in chain.stream(formatted_prompt):
    print(chunk, end="", flush=True)
