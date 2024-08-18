import os
import functools
import operator
import subprocess
from typing import Annotated, Sequence, TypedDict
from typing import Literal
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langchain_experimental.utilities import PythonREPL

# 프로세스 실행을 위한 환경설정 및 파라미터 준비
# LangSmith 사이트에서 아래 내용을 복사해서 .env 파일에 입력
# TAVILY_API_KEY=<your_api_key>
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# LANGCHAIN_API_KEY=<your_api_key>
# LANGCHAIN_PROJECT="<your_project_name>"
load_dotenv()


#################################################################
# 1. Utility 함수 및 LangChain/Agent/LangGraph 관련 함수 정의
#################################################################
# 1-1. 입력한 파라미터를 이용해서 Agent 를 생성하는 함수
def create_agent(llm, tools, system_message: str):
    """Create Researcher agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


# 1-2. Graph node 사이에 전달되는 state 데이터 클래스를 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# 1-3. Agent 에 해당하는 node 를 생성
#      이 함수 자체가 node 에 해당하며, agent 를 실행해주는 executor 이다
#      AgentState 객체인 state 를 입력받고 새로운 state 를 반환한다
#      즉, Node 의 agent 실행 결과는 다음 실행할 node 정보를 담고 있어야 한다.
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)

    print(f"Current agent name = {name}")
    return {
        "messages": [result],
        "sender": name,
    }


# 1-4. Node 간 작업을 조율할 Router 생성
#      Router 에 전달된 state 를 통해 현재 상태를 파악
#      다음 수행할 작업을 string 으로 리턴 (Literal 로 미리 지정된 문자열 중 하나를 사용)
def router(state) -> Literal["call_tool", "__end__", "continue", "to_chart_generator"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        print(f"-----> Router: Calling ToolExecutor, sender={state["sender"]}\n----->")
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        print(f"-----> Router: __end__, sender={state["sender"]}\n----->")
        return "__end__"
    if "to_chart_generator" in last_message.content:
        print(f"-----> Router: Move to ChartGenerator, sender={state["sender"]}\n----->")
        return "to_chart_generator"
    if state["sender"] == "Researcher":
        print(f"-----> Router: Move to ChartGenerator, sender={state["sender"]}\n----->")
        return "to_chart_generator"
    print(f"-----> Router: continue, sender={state["sender"]}\n----->")
    return "continue"


#################################################################
# 2. Agent 에서 사용할 도구들 (Tools)
#################################################################
# 2-1. PythonREPL 인스턴스 생성
# Warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()

# 2-2. 파이썬 코드 실행 도구 정의
@tool
def python_repl(code):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        # Matplotlib 백엔드 설정 추가 (메인 스레드 외부에서 실행 가능하도록)
        setup_code = """
import matplotlib
import os
import sys

if sys.platform.startswith('win'):
    matplotlib.use('TkAgg')  # Windows
elif sys.platform.startswith('darwin'):
    matplotlib.use('MacOSX')  # macOS
else:
    matplotlib.use('TkAgg')  # Linux and others

import matplotlib.pyplot as plt
"""
        # 파이썬 코드 실행
        repl.run(setup_code)
        result = repl.run(code)
    except BaseException as ex:
        print(f"Execution failed. Error: {repr(ex)}")
        return f"Failed to execute. Error: {repr(ex)}"

    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


# 2-3. AI 검색 도구 정의, [.env] 파일에 TAVILY_API_KEY 환경설정 필요
tavily_tool = TavilySearchResults(max_results=5)

# 2-4. ToolExecutor 에서 사용할 Tool list
tools = [tavily_tool, python_repl]


#################################################################
# 3. 그래프 구조 생성 (LLM, agent, node, router, workflow)
#################################################################
# 3-1. LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini")

# 3-2. Researcher Agent 생성
research_agent = create_agent(
    llm,
    [tavily_tool],
    # Agent 의 역할과 tool/agent 와의 연동이 원활하도록 충분히 설명 필요
    system_message="""You are a Research agent.
Your role is to gather accurate and concise data using search tools.
(Use tavily_tool for this job)
You should provide accurate data for the ChartGenerator to use.
When you are ready to launch ChartGenerator, include below string at your response.
'to_chart_generator'
""",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# 3-3. ChartGenerator Agent 생성
chart_agent = create_agent(
    llm,
    [python_repl],
    # Agent 의 역할과 tool/agent 와의 연동이 원활하도록 충분히 설명 필요
    system_message="""You are a ChartGenerator agent.
When you receive data from Researcher, create python codes to draw a chart.
Always use the python_repl tool to execute your code.
After generating the chart, respond with FINAL ANSWER.""",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="ChartGenerator")

# 3-4. Tool node
tool_node = ToolNode(tools)

# 3-5. Workflow(builder) 생성, Node 설정
workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("ChartGenerator", chart_node)
workflow.add_node("ToolExecutor", tool_node)

# 3-6. Conditional Edge 설정
#      conditional edge 는 시작 Node 와 Router, Path map 을 설정
#      시작 노드의 실행 결과가 router 로 전달됨
#      router 에서 다음 실행할 노드를 찾음 (next node string 리턴)
#      path map 에서 다음 실행할 노드를 확인
workflow.add_conditional_edges(
    "Researcher",
    router,
    {
        "continue": "ChartGenerator", # 데이터 작업이 끝나면 chart 를 작성
        "to_chart_generator": "ChartGenerator", # 데이터 작업이 끝나면 chart 를 작성
        "call_tool": "ToolExecutor",
        "__end__": END,
    },
)
workflow.add_conditional_edges(
    "ChartGenerator",
    router,
    {"continue": "ChartGenerator", "call_tool": "ToolExecutor", "__end__": END},
)
workflow.add_conditional_edges(
    "ToolExecutor",
    # Agent 노드는 'sender' 필드값을 state 데이터에 담아서 전송
    # -> ToolExecutor 는 작업이 끝나면 자신을 호출한 sender 를 다시 호출함
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "ChartGenerator": "ChartGenerator",
    },
)
# 3-7. 그래프의 시작 node 를 지정
workflow.add_edge(START, "Researcher")

# 3-8. 그래프 생성
graph = workflow.compile()

#################################################################
# 4. Graph 구조 이미지 저장 및 출력, 그래프 실행
#################################################################
# 4-1. 그래프 구조 이미지 출력
try:
    # 그래프를 PNG 파일로 저장
    png_data = graph.get_graph(xray=True).draw_mermaid_png()

    # 현재 작업 디렉토리에 'graph.png' 파일로 저장
    file_path = os.path.join(os.getcwd(), "graph.png")
    with open(file_path, "wb") as f:
        f.write(png_data)

    print(f"Graph saved as {file_path}")

    # Windows의 기본 이미지 뷰어로 파일 열기
    subprocess.run(["start", file_path], shell=True, check=True)
except Exception as e:
    print(f"An error occurred: {e}")

# 4-2. 그래프 실행
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="""Fetch the UK's GDP data over the past 3 years, 
                then create a line graph of it using Python. 
                Use the Researcher to get the data and call the ChartGenerator to create the graph. 
                The ChartGenerator MUST use the python_repl tool to execute the code 
                and create the chart."""
            )
        ],
    },
    RunnableConfig(recursion_limit=20),  # Maximum number of steps
)

# 4-3. 발생하는 이벤트를 모두 출력 (graph 실행 완료까지 대기하기 위해 아래 코드가 필요)
for s in events:
    # 출력 내용이 무척 많기 때문에 상세 내용은 LangSmith 사이트에서 확인하는 편이 좋다
    # print(s)
    # print("-------------------")
    pass
