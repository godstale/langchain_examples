import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser

# 프로세스 실행을 위한 환경설정 및 파라미터 준비
# LangSmith 사이트에서 아래 내용을 복사해서 .env 파일에 입력
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# LANGCHAIN_API_KEY=<your_api_key>
# LANGCHAIN_PROJECT="<your_project_name>"

load_dotenv()  # .env 파일 로드

# 1. 뉴스 URL
news_url = """https://www.bbc.com/korean/articles/c166p510n79o"""

# 2. 뉴스 스크래핑
loader = WebBaseLoader(
    web_paths=([news_url]),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["bbc-1cvxiy9", "bbc-fa0wmp"]},
        )
    ),
)
news_array = loader.load()
news = news_array[0]

# 3. 요약에 사용할 프롬프트 불러오기
prompt = hub.pull("hellollama/news_summary")

# 4. Ollama 초기화
llm = Ollama(model="llama3.1", temperature=0)

# 5. 프롬프트를 실행할 체인생성
summary_chain = prompt | llm | StrOutputParser()

# 6. 프롬프트에 스크래핑한 기사 원문을 넣어줘야한다
formatted_prompt = {"news": news}

# 7. LLM에 질문
for chunk in summary_chain.stream(formatted_prompt):
    print(chunk, end="", flush=True)
