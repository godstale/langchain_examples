from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 1. 대화내용 저장을 위한 ChatPromptTemplate 설정
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 유능한 기상학자입니다. 답변은 200자 이내로 하세요."),
    MessagesPlaceholder("chat_history"), # 1-1. 프롬프트에 대화 기록용 chat_history 추가
    ("user", "{question}")
])

# 2. Ollama 모델 초기화
llm = Ollama(model="llama3.1")

# 3. 스트림 출력 파서 생성
class CustomStreamOutputParser(StrOutputParser):
    def parse(self, text):
        return text

output_parser = CustomStreamOutputParser()

# 4. chain 연결 (LCEL)
chain = prompt | llm | output_parser

# 5. 채팅 기록 초기화
chat_history = []

# 6. chain 실행 및 결과 출력을 반복
while True:
    # 6-1. 사용자의 입력을 기다림
    user_input = input("\n\n당신: ")
    if user_input == "끝":
        break

    # 6-2. 체인을 실행하고 결과를 stream 형태로 출력
    result = ""
    for chunk in chain.stream({"question": user_input, "chat_history": chat_history}):
        print(chunk, end="", flush=True)
        result += chunk

    # 6-3. 채팅 기록 업데이트
    chat_history.append(("user", user_input))
    chat_history.append(("assistant", result))