from langchain_community.llms import Ollama
# Ollama 연결 (localhost:11434에서 실행 중인 Ollama에 연결)
llm = Ollama(model="llama3.1")
# chain 실행
response = llm.invoke("세계에서 가장 붐비는 항공노선은 어디인가?")
# 결과 출력
print(response)