import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.agents.react.agent import create_react_agent
from langchain.chains.llm_math.base import LLMMathChain
from langchain.agents import Tool
from langchain.agents.agent import AgentExecutor


load_dotenv()

# 1. 프롬프트 템플릿 준비
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following questions as best you can.
    You have access to the following tools:
    {tools}

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    ============================= Chat history =============================
    {chat_history}
    ================================ Human Message =================================
    Question: {question}
    ============================= Messages Placeholder =============================
    Thought:{agent_scratchpad}
    """
)

# 2. LLM 초기화
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-mini",
    openai_api_key=openai_api_key,
)

# 3. Agent 도구 초기화
# 3-1. DuckDuckGo 검색 도구 초기화
search_tool = Tool(
    name="Search", func=DuckDuckGoSearchRun().run, description="DuckDuckGo 웹 검색 도구"
)
# 3-2. Math 계산 도구 초기화
math_tool = Tool.from_function(
    name="Calculator",
    func=LLMMathChain.from_llm(llm=llm).run,
    description="""Useful for when you need to answer questions about math.
        This tool is only for math questions and nothing else.
        Only input math expressions.""",
)
# 3-3. 도구 리스트 생성
tools = [search_tool, math_tool]

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
    if user_input in ("끝", "q", "exit"):
        break

    # 5-2. Agent 실행
    result = agent_executor.invoke(
        {"question": user_input, "chat_history": chat_history}
    )
    # 5-3. 응답이 dictionary 로 전달되므로 output 만 추출해서 출력
    output_text = result["output"]
    print(f"\nAI --->\n{output_text}")

    # 5-4. 대화내역 저장
    chat_history.append(f"user: {user_input}")
    chat_history.append(f"assistant: {output_text}")
