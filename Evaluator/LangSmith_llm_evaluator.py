import os
import re
from typing import List
from dotenv import load_dotenv
from langchain import hub
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from langchain.smith import RunEvalConfig
from langchain.smith import run_on_dataset
from langchain.evaluation import CriteriaEvalChain


# 1. 프로세스 실행을 위한 환경설정 및 파라미터 준비
# LangSmith 사이트에서 아래 내용을 복사해서 .env 파일에 입력
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# LANGCHAIN_API_KEY=<your_api_key>
# LANGCHAIN_PROJECT="<your_project_name>"

load_dotenv()  # .env 파일 로드

################################################################
# 1. 테스트 준비 단계
################################################################
# 1-1. LangSmith project, dataset 이름으로 사용할 문자열
project_name = os.getenv("LANGCHAIN_PROJECT")

# 1-2. 테스트용 input, output 데이터 셋 준비
news_array = [
    """아프리카 펭귄을 구하기 위한 투쟁사진 출처, Christian Parkinson/BBC기사 관련 정보기자, 제니 힐 기자,  BBC 뉴스 Reporting from  웨스턴 케이프 2024년 5월 12일매년 남아프리카공화국 
과 나미비아 등 아프리카 펭귄의 수가 줄고 있다. 곧 아예 없어질 수도 있다고 한다. 과학자들은 이 종이 매년 약 8%씩 감소하고 있다고 밝혔다.베티스 베이의 펭귄들이 물가를 따라 뛰어다니며 짧은 쉰 목
소리로 서로를 부른다.펭귄들은 유쾌하게 돌아다니고 있지만 한 사람의 얼굴에는 근심이 가득했다."물 가까이에 있는 이 펭귄은 꽤 여유가 없어 보여요. 살이 많이 없다는 걸 알 수 있죠."버드라이프 남아 
프리카공화국 해조류 보호 연구원인 맥인니스 박사는 이 지역 내 줄어드는 펭귄들을 추적 및 감시하는 일을 하고 있다.지난 세기 동안 아프리카 펭귄은 99% 감소했다."지금의 감소율이 지속된다면, 2035년 
까지 종의 멸종을 목격할 수도 있다"고 맥인니스 박사는 경고한다.그것이 바로 버드라이프 남아프리카공화국 해안 조류 보호재단(SANCCOB)이 이 곳에서 처음으로 법적 조치를 취하고 있는 이유다.이들은 이
 국가가 멸종 위기에 처한 종을 충분히 보호하지 못했다고 주장한다. "생물 다양성 법센터"의 케이트 핸들리는 이는 해당 국가의 법적 의무라고 강조했다.사진 출처, Christian Parkinson/BBC사진 설명, 펭
귄이 의존하는 정어리와 멸치의 개체수는 감소하고 있다대부분의 아프리카 펭귄들은 아프리카 남서부 해안을 따라 일곱 군데에서 사는 것으로 추정된다. 약 8,750쌍이다.펭귄은 고정된 흑색 줄이 몸쪽을 따
라 빠지는 짧고 튼튼한 새다. 그들은 사진을 찍는 사람들에게 아무 영향도 받지 않는 것처럼 보이지만, 햇볕에 누워있거나 알을 지키는 동안 그들은 불안한 시선으로 사람들을 지키보고 있다.그들은 자연적
인 포식자; 특히 물개와 특정 종류의 갈매기에도 취약하다. 하지만 진짜 적은 인간이다.현재 중단된 그노(펭귄굴에 쌓인 배설물) 수확 실천은 그들의 서식지를 파괴했다. 게다가 기후 변화가 문제를 악화시
키고 있다. 폭풍과 홍수는 펭귄들을 위협하고 바다의 흐름과 온도가 변화함에 따라 펭귄들이 먹이에 접근하기가 점점 더 어려워지고 있다.그리고 펭귄이 의존하는 마른 멸치와 정어리는 상업 어업에도 중요
하다.남아프리카공화국 정부는 대규모 그물을 사용해 물고기 무리를 잡는 소위 '축어채 어업 선박'의 활동을 제한하려고 노력해왔다. 하지만 이것은 불안정한 문제를 야기한다.지난 15년 동안 어업지역의  
실험적 폐쇄, 어업 업계와 보전가들 간의 길고 복잡한 협상, 국제 전문가 패널의 의견 등이 바로 그것이다.하지만 펭귄 수는 여전히 감소하고 있다.버드라이프 남아프리카공화국 해안 조류 보호재단(SANCCOB)은 현재 이뤄지고 있는 펭귄 주변에서의 어업 금지 조치가 충분히 확장되지 않았거나 올바른 위치에 있지 않다고 주장한다.그들의 변호사들은 "생물학적으로 의미 있는" 조치의 즉각적인 시행을 요구하고
 있다.새들이 충돌하지 않는 풍력 발전단지를 조성하는 방법2024년 5월 4일영원히 사라져버린 메가배트2024년 4월 27일그 많던 오징어는 다 어디로 갔을까2024년 3월 22일사진 출처, Christian Parkinson/BBC사진 설명, 버드라이프 남아프리카공화국 해조류 보호 연구원인 맥인니스 박사그러나 해안을 따라 작은 항구에서 어부들이 바다로 나가기 전에 물고기를 내리는 동안 걱정과 분노가 존재한다.이곳 사람들
은 자신들이 책임을 져야 한다는 지적을 거부했다."우리는 문제의 일부분입니다" 많은 어부들을 대표하는 남아프리카공화국 펠라직 어업협회의 부회장인 샤메라 다니엘스는 말했다."포식자인 물개, 상어는 
물론 석유 및 가스 탐사, 소음 오염도 문제입니다."그는 또 현재의 제한이 이미 수백만 달러의 비용과 수백 개의 직장을 어업업에 야기했다고 주장했다. 추가적인 폐쇄는 여기서 많은 사람들이 의지하는 산
업에 더 많은 고통을 초래할 것이라는 말이다.사진 출처, Christian Parkinson/BBC사진 설명, 펭귄은 관광상품이 되었지만, 더 이상 그 곳에 존재하지 않을지도 모른다앞으로 길고 긴 법적 과정이 기다리고
 있다.보존 법률가인 핸들리는 현실적으로 시간이 부족하다고 털어놓는다."아프리카 펭귄을 구하기 위한 모든 단계는 우리가 실제로 이러한 법정 청문회를 시간 내에 얻을 수 있는 가능성이 얼마 없더라도 
시도해야 한다는 것입니다."첫 번째 법적 청문회가 언제 개최될지는 아직 확실하지 않다. 남아프리카공화국 정부는 이에 대한 논평을 하지 않았다.하지만 우리는 이러한 조치들이 아프리카의 죽어가는 펭귄
들에게는 이미 너무 늦었을 수 있다는 것을 기억해야 한다.추가 취재: 가비 콜렌소관련 기사 더 보기에베레스트 등반? 이제는 당신의 배설물을 가져와야 한다2024년 2월 12일북극 얼음 녹아 육지로 떠밀린 
북극곰들...적응 못해 굶주림 직면2024년 2월 14일'구름 씨앗'이란 무엇이며, 이번 두바이 홍수의 원인은?2024년 4월 18일'양식 연어' 대량 폐사가 급증하는 이유2024년 3월 10일아마존 최악의 가뭄…'한번 
도 본 적 없는 모습입니다'2023년 12월 26일'기후 재앙 마지노선 1.5℃ 돌파 가능성 커졌다'2023년 10월 7일""",
]
summary_list = [
    """<news>
  <title>아프리카 펭귄을 구하기 위한 투쟁</title>
  <author>제니 힐 기자, BBC 뉴스</author>
  <date>2024년 5월 12일</date>
  <summary>아프리카 펭귄의 수가 매년 약 8%씩 감소하고 있으며, 현재 남아프리카공화국과 나미비아에서 이 종의 멸종 위기가 심각해지고 있다. 과학자들은 2035년까지 멸종할 수 있다고 경고하고 있으며, 이는 인간의 활동과 기후 변화가 주요 원인으로 지목되고 있다. 버드라이프 남아프리카공화국 해안 조류 보호재단은 정부가 충분한 보호 조치를 취하지 않고 있다고 주장하며 법적 조치를 취하고 있다. 그
러나 어업계는 이러한 조치가 산업에 부정적인 영향을 미칠 것이라고 반발하고 있다. 향후 법적 과정이 진행될 예정이지만, 이미 늦었을 수 있다는 우려가 제기되고 있다.</summary>
</news>""",
]

# 1-3. LangSmith client 생성
client = Client()

# 1-4. input, output 데이터를 저장할 dataset 생성 또는 가져오기
try:
    ds = client.create_dataset(
        dataset_name=project_name,
        description="An example dataset of news and summary",
    )
except Exception as e:
    # 이미 존재하는 경우, 기존 dataset을 가져옵니다
    existing_datasets = list(client.list_datasets(dataset_name=project_name))
    if existing_datasets:
        ds = existing_datasets[0]
    else:
        print(f"----- Failed to create or retrieve dataset: {e}")

# 1-5. 데이터셋에 example (input-output) 추가
try:
    client.create_examples(
        inputs=[{"news": news} for news in news_array],
        outputs=[{"output": summary} for summary in summary_list],
        dataset_id=ds.id,
    )
except Exception as e:
    print(f"----- Failed to add example: {e}")

# 1-6. 뉴스 요약 PROMPT Template 생성
summary_prompt = hub.pull("hellollama/news_summary")

# 1-7. 뉴스 요약에 사용할 LLM 초기화
llm = Ollama(model="llama3.1", temperature=0)

# 1-8. 뉴스 요약 체인 (테스트용 체인)
summary_chain = summary_prompt | llm | StrOutputParser()

################################################################
# 2. 평가 단계
################################################################
# 2-1. 뉴스 요약 결과를 평가할 LLM 초기화
openai_api_key = os.getenv("OPENAI_API_KEY")  # OpenAI API 키 가져오기
evaluator_llm = ChatOpenAI(
    temperature=0,  # 정확성 <-> 창의성 (0.0 ~ 2.0)
    max_tokens=1024 * 10,  # 최대 토큰수
    model_name="gpt-4o-mini",  # 모델명
    openai_api_key=openai_api_key,  # API Key 설정
)

print("----- Set evaluation configurations")


# 2-2. 평가 결과에서 score 추출해서 하나의 리포트로 만들어주는 함수
def batch_evaluator(runs: List[Run], examples: List[Example]):
    """evaluator 실행 결과를 취합해서 점수로 환산"""
    print("\n\n----- Run batch_evaluator")
    run_ids = []
    for run in runs:
        run_ids.append(str(run.id))
    feedback_list = client.list_feedback(run_ids=run_ids)
    average_score = 0
    feedback_size = 0
    for feedback in feedback_list:
        # print(f"feedback = {feedback.comment}")
        pattern = r"<(\w+)>(\d+)</\1>"  # 정규 표현식 패턴
        matches = re.findall(pattern, feedback.comment)  # 값 추출
        results = {}  # 결과를 저장할 딕셔너리
        # 추출된 값들을 딕셔너리에 저장
        for tag, value in matches:
            results[tag] = int(value)
        # 결과 출력
        for tag, value in results.items():
            print(f"{tag}: {value}")
        average_score += results["score"]
        feedback_size += 1
    average_score = average_score / feedback_size
    return EvaluationResult(key="score", score=average_score)


# 2-3. 평가 기준 정의
criteria = {
    "clarity": "The explanation should be clear and easy to understand. Express score as a number between 1-10. Write score between <clarity></clarity> tag",  # 명료성
    "accuracy": "The information provided should be factually correct. Express score as a number between 1-10. Write score between <accuracy></accuracy> tag",  # 사실성
    "conciseness": "The explanation should be concise and to the point. Express score as a number between 1-10. Write score between <conciseness></conciseness> tag",  # 정확성
    "score": "Average of the clarity, accuracy, and conciseness score as a number between 1-10. Write average score between <score></score> tag",  # 종합 평가 점수 (평균)
}

# 2-4. 평가용 프롬프트
evaluation_prompt = hub.pull("hellollama/summary_evaluator")

# 2-5. 평가용 체인 - evaluator 생성
evaluator = CriteriaEvalChain.from_llm(
    llm=evaluator_llm,
    criteria=criteria,
    prompt=evaluation_prompt,
)

# 2-6. 평가용 환경설정
evaluation_config = RunEvalConfig(
    # 주의!! LLM 지정없이 아래와 같이 실행하면 GPT-4o 로 실행됨
    # evaluators=[RunEvalConfig.Criteria(criteria)],
    # Custom evaluator 로 실행
    custom_evaluators=[evaluator],
    batch_evaluators=[batch_evaluator],
)

# 2-7. 평가용 환경설정을 추가해서 데이터셋 데이터에 뉴스 요약 체인을 실행
chain_results = run_on_dataset(
    dataset_name=project_name,
    llm_or_chain_factory=summary_chain,
    evaluation=evaluation_config,
    verbose=True,
    client=client,
    # Project metadata communicates the experiment parameters,
    # Useful for reviewing the test results
    project_metadata={
        "env": "Ollama",
        "model": "llama3.1",
        "prompt": "hellollama/news_summary",
    },
)

# 결과를 저장할 리스트들 초기화
input_news_list = []
output_string_list = []
run_id_list = []
reference_output_list = []

# results의 각 항목에 대해 처리
for result_key, result_value in chain_results["results"].items():
    # 1. results -> input -> news 문자열
    input_news_list.append(result_value["input"]["news"])
    # 2. results -> output 문자열
    output_string_list.append(result_value["output"])
    # 3. result -> run_id
    run_id_list.append(result_value["run_id"])
    # 4. result -> reference -> output 문자열
    reference_output_list.append(result_value["reference"]["output"])

# 2-8. 결과 출력
print("Input News List:", input_news_list)
print("Output String List:", output_string_list)
print("Run ID List:", run_id_list)
print("Reference Output List:", reference_output_list)
print("aggregate_metrics:", chain_results["aggregate_metrics"])
