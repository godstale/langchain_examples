import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent

# 프로세스 실행을 위한 환경설정 및 파라미터 준비
# LangSmith 사이트에서 아래 내용을 복사해서 .env 파일에 입력
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# LANGCHAIN_API_KEY=<your_api_key>
# LANGCHAIN_PROJECT="<your_project_name>"

load_dotenv()

# 1. 프롬프트 템플릿 준비
#    {tools} : 사용가능한 도구를 Agent 에게 알려주기 위해 필요한 파라미터 (필수)
#    {tool_names} : 다음 작업(Action)에 필요한 도구들을 표시하는 파라미터 (필수)
#    {agent_scratchpad} : Agent 가 수행해야 하는 작업의 내용을 표시 (필수)
prompt = ChatPromptTemplate.from_template(
    """당신은 유능한 기상학자입니다. 필요한 경우 다음 도구들을 사용할 수 있습니다: 
    {tools}

    Use the following format:
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    ============================= Chat history =============================
    {chat_history}
    ================================ Human Message =================================
    Question: {question}
    ============================= Messages Placeholder =============================
    Thought: {agent_scratchpad}
    """
)

# 2. LLM 초기화
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    temperature=0,
    max_tokens=2048,
    model_name="gpt-4o-mini",
    openai_api_key=openai_api_key,
)

# 3. DuckDuckGo 검색 도구 초기화
search = DuckDuckGoSearchRun()
tools = [search]

# 4. agent 생성
agent = create_react_agent(llm, tools, prompt)
# 4-1. agent_executor 생성
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 이전 챗을 기억하기 위해 chat_history 사용
chat_history = []

# 5. 챗 실행
while True:
    # 5-1. 사용자 질문을 입력 받음
    user_input = input("\n\n당신 (종료 q): ")
    if user_input in ("끝", "종료", "q", "exit"):
        break

    # 5-2. Agent 실행
    result = agent_executor.invoke(
        {"question": user_input, "chat_history": chat_history}
    )
    # 5-3. 응답이 dictionary 로 전달되므로 output 만 추출해서 출력
    output_text = result["output"]
    print(f"\nAI --->\n{output_text}")

    # 대화내역 저장
    chat_history.append(f"user: {user_input}")
    chat_history.append(f"assistant: {output_text}")
