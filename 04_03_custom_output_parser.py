from typing import Iterable
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableGenerator
from langchain_core.messages import AIMessageChunk

# 1. 대화내용 저장을 위한 ChatPromptTemplate 설정
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 유능한 기상학자입니다. 답변은 200자 이내로 하세요."),
    MessagesPlaceholder("chat_history"), # 1-1. 프롬프트에 대화 기록용 chat_history 추가
    ("user", "{question}")
])

# 2. Ollama 모델 초기화
llm = Ollama(model="llama3.1")

# 3. 스트림 출력 파서 생성 🌪️
def replace_word_with_emoji(text: str) -> str: # 문자열에서 태풍을 이모지로 바꿔주는 함수
    return text.replace("태풍", "🌪️")

def streaming_parse(chunks: Iterable[AIMessageChunk]) -> Iterable[str]: # type: ignore
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        while " " in buffer: # 속도가 느린 컴퓨터에서 실행하는 경우 단어가 완성될 때까지 모아서 처리
            word, buffer = buffer.split(" ", 1)
            yield replace_word_with_emoji(word) + " "
    if buffer:
        yield replace_word_with_emoji(buffer)

streaming_parser = RunnableGenerator(streaming_parse)

# 4. chain 연결 (LCEL)
chain = prompt | llm | streaming_parser

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