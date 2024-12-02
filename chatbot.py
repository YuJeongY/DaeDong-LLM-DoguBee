
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, DataFrameLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
import pandas as pd

# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

# 폴더 경로 설정
folder_path = "./data"  # 분석할 파일이 저장된 폴더 경로
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# PDF 문서 로드 함수
def load_pdf_with_metadata(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split(text_splitter)
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["page"] = doc.metadata.get("page", "Unknown")
    return documents

# 엑셀 문서 로드 함수
def load_excel_with_metadata(file_path):
    documents = []
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        loader = DataFrameLoader(df, page_content_column=df.columns[0])
        sheet_docs = loader.load_and_split(text_splitter)
        for doc in sheet_docs:
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["sheet_name"] = sheet_name
            doc.metadata["cell_range"] = f"A1:{df.columns[-1]}{len(df)}"  # 추가 셀 범위 정보
        documents.extend(sheet_docs)
    return documents


# CSV 문서 로드 함수
def load_csv_with_metadata(file_path):
    documents = []
    df = pd.read_csv(file_path)
    loader = DataFrameLoader(df, page_content_column=df.columns[0])
    csv_docs = loader.load_and_split(text_splitter)
    for doc in csv_docs:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["cell_range"] = f"A1:{df.columns[-1]}{len(df)}"  # 추가 셀 범위 정보
    documents.extend(csv_docs)
    return documents

# 폴더 내 모든 문서를 로드

def load_documents_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".pdf"):
            documents.extend(load_pdf_with_metadata(file_path))
        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            documents.extend(load_excel_with_metadata(file_path))
        elif file_name.endswith(".csv"):
            documents.extend(load_csv_with_metadata(file_path))
    return documents



# 에이전트와 대화하는 함수
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    response = result['output']  # 명시적으로 출력 키를 처리
    return response

# 세션 기록 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]

# 대화 내용 출력하는 함수
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])


# 모든 문서 로드
all_docs = load_documents_from_folder(folder_path)


# FAISS 인덱스 설정 및 생성
vector = FAISS.from_documents(all_docs, OpenAIEmbeddings())
retriever = vector.as_retriever()

# 도구 정의
retriever_tool = create_retriever_tool(
    retriever,
    name="retriever_tool",
    description="Use this tool to search information from the pdf document"
)

# Streamlit 메인 코드
def main():
    # 페이지 설정
    st.set_page_config(page_title="하도급 관련 상담 전문가 DoguBee", layout="wide", page_icon="🐝")

    st.image('DoguBee.png', width=450)
    st.markdown('---')
    st.title("안녕하세요! 하도급 관련 상담 전문가 'DoguBee' 입니다")  # 시작 타이틀

    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}


# return retriever_tool
    tools = [retriever_tool]

    # LLM 설정
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Prompt 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                    "You are an expert in subcontracting law. "
                    "Use sophisticated and credible language to maintain authority. "
                    "When providing examples or case-related answers, organize them clearly by separating each case for better readability."
                    "Always end your responses with a phrase encouraging additional questions, such as 'Feel free to ask if you have further questions.'"
                    "Do not answer questions unrelated to subcontracting law. Instead, respond with a message like: 'This question is not related to subcontracting law. Please ask only about matters related to subcontracting law."
                    
                    "Please always reply in Korean and kindly."

            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 에이전트 생성 (initialize_agent 대신 create_tool_calling_agent 사용)
    agent = create_tool_calling_agent(llm, tools, prompt)

    # AgentExecutor 정의
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 사용자 입력 처리
    user_input = st.chat_input('질문이 무엇인가요?')

    if user_input:
        session_id = "default_session"
        session_history = get_session_history(session_id)

        if session_history.messages:
            previous_messages = [{"role": msg['role'], "content": msg['content']} for msg in session_history.messages]
            response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(previous_messages), agent_executor)
        else:
            response = chat_with_agent(user_input, agent_executor)

        # 메시지를 세션에 추가
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # 세션 기록에 메시지를 추가
        session_history.add_message({"role": "user", "content": user_input})
        session_history.add_message({"role": "assistant", "content": response})

        # 대화 내용 출력
        print_messages()



if __name__ == "__main__":
    main()
