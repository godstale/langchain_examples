from langchain_community.llms import Ollama
import sys

llm = Ollama(temperature=0, model="llama3.1")

question = "세계에서 가장 붐비는 항공노선 4개를 알려줘"

summary_query = f"""
아래 질문에 대한 대답을 할 때 IATA 자료를 참고해. 대답은 아래 형식으로 만들어줘. 마지막에는 출처를 적어줘.
형식:
1. 대답 1
2. 대답 2

질문: {question}
"""

# 스트리밍 출력
for chunk in llm.stream(summary_query):
    sys.stdout.write(chunk)
    sys.stdout.flush()

print()  # 마지막에 줄바꿈을 추가