import os
import bs4
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from langchain_openai import ChatOpenAI

# 1. 프로세스 실행을 위한 환경설정 및 파라미터 준비
# LangSmith 사이트에서 아래 내용을 복사해서 .env 파일에 입력
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# LANGCHAIN_API_KEY=<your_api_key>
# LANGCHAIN_PROJECT="<your_project_name>"

load_dotenv()  # .env 파일 로드

# 1-1. LangSmith project, dataset, annotation queue 이름으로 사용할 문자열
project_name = os.getenv("LANGCHAIN_PROJECT")

# 1-2. 테스트 대상이 되는 뉴스 URLs
news_urls = [
    """https://www.bbc.com/korean/articles/c166p510n79o""",
    """https://www.bbc.com/korean/articles/cjqe104j8l0o""",
    """https://www.bbc.com/korean/articles/c2lewq29ww7o""",
]
data_cache = {}  # 데이터 임시 저장용 캐시 생성

# 1-3. langsmith client 생성
client = Client()

# 1-4. 뉴스 요약 PROMPT Template 생성
prompt = PromptTemplate.from_template(
    """당신은 뉴스 기사를 요약, 정리하는 AI 어시스턴트입니다. 
당신의 임무는 주어진 뉴스(news_text)에서 기사제목(title), 작성자(author), 작성일자(date), 요약문(summary) - 4가지 항목을 추출하는 것입니다.
결과는 한국어로 작성해야합니다. 뉴스와 관련성이 높은 내용만 포함하고 추측된 내용을 생성하지 마세요.

<news_text>
{news}
<news_text>

요약된 결과는 아래 형식에 맞춰야합니다.
<news>
  <title></title>
  <author></author>
  <date></date>
  <summary></summary>
</news>
"""
)
data_cache["current_prompt"] = prompt

# 1-5. PROMPT 최적화용 Template 생성
optimizer_prompt = PromptTemplate.from_template(
    """당신은 AI 프롬프트 전문가입니다. 
아래 뉴스 요약 프롬프트(<prompt>)가 있습니다.
<prompt>
{current_prompt}
</prompt>

AI 가 프롬프트를 사용하여 기사 원문(<originaltext>)을 요약한 결과(<results>)는 아래와 같습니다.
<originaltext>
{original_text} 
</originaltext>

<results>
{summary}
</results>

그리고 결과의 만족도를 평가한 피드백(<feedbacks>)이 주어집니다.
<feedbacks>
{human_feedback}
</feedbacks>

당신은 이 결과를 참고해서 기존의 프롬프트(<prompt>)를 개선해야 합니다.
개선된 프롬프트는 <newprompt></newprompt> 태그 사이에 넣어주세요.
"""
)

# 1-5. 뉴스 스크래핑 (Document Loading)
loader = WebBaseLoader(
    web_paths=(news_urls),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["bbc-1cvxiy9", "bbc-fa0wmp"]},
        )
    ),
)
news_array = loader.load()

# 1-6. 뉴스 원문을 저장할 dataset 생성 또는 가져오기
try:
    ds = client.create_dataset(dataset_name=project_name)
    print(f"----- New dataset created: {ds}")
except Exception as e:
    # 이미 존재하는 경우, 기존 dataset을 가져옵니다
    existing_datasets = list(client.list_datasets(dataset_name=project_name))
    if existing_datasets:
        ds = existing_datasets[0]
        print(f"----- Using existing dataset: {ds}")
    else:
        print(f"----- Failed to create or retrieve dataset: {e}")

# 1-7. 사용할 LLM 초기화
llm = Ollama(model="llama3.1", temperature=0)

openai_api_key = os.getenv("OPENAI_API_KEY")  # OpenAI API 키 가져오기
optimizer_llm = ChatOpenAI(
    temperature=0,  # 정확성 <-> 창의성 (0.0 ~ 2.0)
    max_tokens=2048,  # 최대 토큰수
    model_name="gpt-4o-mini",  # 모델명
    openai_api_key=openai_api_key,  # API Key 설정
)

################################################################
# 2. 로딩한 뉴스 데이터 1개씩 처리 -> 모두 처리할 때까지 반복
#    실험(experiment) 단계: 뉴스 데이터를 이용해서 테스트, 피드백 준비
################################################################
for news in news_array:
    # 2-1. 중간중간 생성되는 데이터를 담는 변수 선언
    data_cache["news"] = news
    example_uuid = str(uuid.uuid4())

    # 2-2. 앞서 생성한 데이터셋에 example 추가
    try:
        client.create_examples(
            ids=[example_uuid],
            inputs=[{"news": news.page_content}],
            dataset_id=ds.id,
        )
        print(f"----- Added new examples successfully ({example_uuid})")
    except Exception as e:
        print(f"----- Failed to add example: {e}")

    # 2-3. 프롬프트를 실행할 체인생성
    summary_chain = prompt | llm | StrOutputParser()

    # 2-4. dataset 에 추가한 뉴스 기사(example)에 뉴스 요약 chain 적용
    print("----- Run chain on new example")
    res = client.run_on_dataset(
        dataset_name=project_name,
        llm_or_chain_factory=summary_chain,
    )
    # 2-5. run_id, 요약 결과 추출
    run_ids = [result["run_id"] for result in res["results"].values()]
    summary = [result["output"] for result in res["results"].values()]

    # 2-6. 기존 annotation queue 검색, 없으면 새로 생성
    existing_queues = list(client.list_annotation_queues(name=project_name))
    if existing_queues:
        # 이미 존재하는 queue 반환
        q = existing_queues[0]
        print(f"\n----- Using existing annotation queue: {q.name}")
    else:
        # 새로운 queue 생성
        q = client.create_annotation_queue(name=project_name)
        print(f"\n----- Created new annotation queue: {q.name}")

    # 2-7. feedback 받기 위해 run_on_dataset 결과를 annotation queue에 추가
    if example_uuid and res:
        try:
            # run 결과를 annotation queue에 추가
            client.add_runs_to_annotation_queue(queue_id=q.id, run_ids=run_ids)
            print(f"----- Added runs to the annotation queue")
        except Exception as e:
            print(f"----- Failed to add runs to the annotation queue: {e}")
    else:
        print("----- No results to add to the annotation queue")

    # 2-8. 평가자의 피드백이 끝날때까지 대기
    while True:
        user_input = input("\n\nLangSmith 에서 피드백을 완료해주세요 [y/q]: ")
        if user_input in ("q", "Q", "exit"):
            quit()  # 프로그램 즉시 종료
        elif user_input in ("y", "Y"):
            break

    #############################################################################
    # 3. 평가(feedback) 단계
    # 평가자 --> LangSmith 사이트에서 (https://smith.langchain.com/)
    # annotation queue 를 확인하면서 feedback 추가
    #############################################################################

    # run_id로 조회하면 run_id 가 동일한 여러개의 feedback이 생성됨
    # 평가 항목별로(correctness, score, comment) feedback이 별도로 존재
    # 따라서 run_id로 그룹화해서 리턴해야함
    def combine_feedback_by_run_id(feedback_list):
        new_feedbacks = {}
        for item in feedback_list:
            run_id = str(item.run_id)
            if run_id in new_feedbacks:
                selected_feedback = new_feedbacks[run_id]
            else:
                selected_feedback = {"correctness": 0, "score": 0, "comment": ""}
                new_feedbacks[run_id] = selected_feedback

            key = item.key
            if key == "correctness":
                selected_feedback["correctness"] = item.score
            elif key == "score":
                selected_feedback["score"] = item.score
            elif key == "note":
                selected_feedback["comment"] = item.comment

        print(f"new_feedbacks = {new_feedbacks}")

        # XML 형식으로 변환
        result = []
        for _key, value in new_feedbacks.items():
            feedback_xml = f"""<feedback>
                <correctness>{value['correctness']}</correctness>
                <score>{value['score']}/5.0</score>
                <comment>{value['comment']}</comment>
                </feedback>"""
            result.append(feedback_xml)
            print(f"feedback_xml = {feedback_xml}\n\n")

        return result

    #############################################################################
    # 4. 프롬프트 최적화(optimization) 단계
    # annotation queue 에서 feedback 추출 -> 피드백을 바탕으로 프롬프트 최적화
    #############################################################################
    # 4-1. 피드백을 가져와서 텍스트로 변환
    print("Getting feedbacks -------------------------------------------------")
    res = client.list_feedback(run_ids=run_ids)
    print("Parsing feedbacks -------------------------------------------------")
    feedback_list = combine_feedback_by_run_id(res)

    # 4-2. 현재 프롬프트, 기사 원문, 요약 결과, 피드백 데이터를 사용해서
    #   프롬프트 최적화 chain 실행
    optimizer = optimizer_prompt | optimizer_llm | StrOutputParser()
    print("Optimize prompt -------------------------------------------------")
    optimized = optimizer.invoke(
        {
            "current_prompt": data_cache["current_prompt"],
            "original_text": data_cache["news"],
            "summary": summary,
            "human_feedback": "\n\n".join(feedback_list),
        }
    )
    print(f"Optimized = \n{optimized}\n")

    # 4-3. 최적화 된 프롬프트를 추출 후 출력
    try:
        new_prompt = optimized.split(f"<newprompt>")[1].split(f"</newprompt>")[0]
    except Exception as e:
        new_prompt = data_cache["current_prompt"]
    print(f"\nNew prompt = \n{new_prompt}")

    # 4-4. 새로운 프롬프트를 저장하고 다음 뉴스 기사를 처리
    while True:
        user_input = input("\n\n다음 예제를 진행할까요? [y/q]: ")
        if user_input in ("q", "Q", "exit"):
            quit()  # 프로그램 즉시 종료
        elif user_input in ("y", "Y"):
            break

    # 앞서 사용한 example 삭제
    # -> run_on_examples 가 모든 example 에 대해 실행되는 문제 때문
    client.delete_example(example_id=example_uuid)

    # 새로운 프롬프트를 사용해서 루프 진행
    data_cache["current_prompt"] = new_prompt

# 프롬프트 최적화 과정 종료
print("\n모든 테스트 샘플을 처리하였습니다. \n")
