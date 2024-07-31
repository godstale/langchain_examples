from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template(
    "당신은 유능한 기상학자입니다. 다음 질문에 답해주세요. <질문>: {question}"
)

# 2. Ollama 모델 초기화
llm = Ollama(model="llama3.1")

# 3. 스트림 출력 파서 생성
class CustomStreamOutputParser(StrOutputParser):
    def parse(self, text):
        return text

output_parser = CustomStreamOutputParser()

# 4. chain 연결 (LCEL)
chain = prompt | llm | output_parser

# 5. chain 실행 및 결과 출력
for chunk in chain.stream({"question": "크기에 따른 태풍의 분류 방법을 알려주세요."}):
    print(chunk, end="", flush=True)
