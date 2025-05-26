import streamlit as st 
import chat
import json
import traceback
import logging
import sys
import mcp_config
import cost_analysis as cost

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("streamlit")

# title
st.set_page_config(page_title='Agent', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

mode_descriptions = {
    "일상적인 대화": [
        "대화이력을 바탕으로 챗봇과 일상의 대화를 편안히 즐길수 있습니다."
    ],
    "RAG": [
        "Bedrock Knowledge Base를 이용해 구현한 RAG로 필요한 정보를 검색합니다."
    ],
    "Agent": [
        "MCP를 활용한 Agent를 이용합니다. 왼쪽 메뉴에서 필요한 MCP를 선택하세요."
    ],
    "Agent (Chat)": [
        "MCP를 활용한 Agent를 이용합니다. 채팅 히스토리를 이용해 interative한 대화를 즐길 수 있습니다."
    ],
    "번역하기": [
        "한국어와 영어에 대한 번역을 제공합니다. 한국어로 입력하면 영어로, 영어로 입력하면 한국어로 번역합니다."        
    ],
    "비용 분석": [
        "Cloud 사용에 대한 분석을 수행합니다."
    ]
}

with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Amazon Bedrock을 이용해 다양한 형태의 대화를 구현합니다." 
        "여기에서는 MCP를 이용해 Agent를 구현합니다." 
        "또한 번역 기능을 사용할 수 있습니다."
        "주요 코드는 LangChain과 LangGraph를 이용해 구현되었습니다.\n"
        "상세한 코드는 [Github](https://github.com/kyopark2014/mcp-agent)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")
    
    # radio selection
    mode = st.radio(
        label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "RAG", "Agent", "Agent (Chat)", "번역하기", "비용 분석"], index=2
    )   
    st.info(mode_descriptions[mode][0])
    
    # mcp selection
    mcp = ""
    if mode=='Agent' or mode=='Agent (Chat)':
        # MCP Config JSON input
        st.subheader("⚙️ MCP Config")

        # Change radio to checkbox
        mcp_options = [
            "default", "code interpreter", "aws document", "aws cli", 
            "aws cloudwatch", "aws storage", "image generation", "aws diagram",
            "knowledge base", "tavily", "perplexity", "ArXiv", "wikipedia", 
            "filesystem", "terminal", "text editor", "playwright", "사용자 설정"
        ]
        mcp_selections = {}
        default_selections = ["default", "tavily", "aws cli", "code interpreter"]

        with st.expander("MCP 옵션 선택", expanded=True):            
            # 2개의 컬럼 생성
            col1, col2 = st.columns(2)
            
            # 옵션을 두 그룹으로 나누기
            mid_point = len(mcp_options) // 2
            first_half = mcp_options[:mid_point]
            second_half = mcp_options[mid_point:]
            
            # 첫 번째 컬럼에 첫 번째 그룹 표시
            with col1:
                for option in first_half:
                    default_value = option in default_selections
                    mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)
            
            # 두 번째 컬럼에 두 번째 그룹 표시
            with col2:
                for option in second_half:
                    default_value = option in default_selections
                    mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)
        
        if not any(mcp_selections.values()):
            mcp_selections["default"] = True

        if mcp_selections["사용자 설정"]:
            mcp_info = st.text_area(
                "MCP 설정을 JSON 형식으로 입력하세요",
                value=mcp,
                height=150
            )
            logger.info(f"mcp_info: {mcp_info}")

            if mcp_info:
                mcp_config.mcp_user_config = json.loads(mcp_info)
                logger.info(f"mcp_user_config: {mcp_config.mcp_user_config}")
        
        mcp = mcp_config.load_selected_config(mcp_selections)
        # logger.info(f"mcp: {mcp}")

    # model selection box
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        ('Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3.0 Sonnet', 'Claude 3.5 Haiku'), index=1
    )

    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'
    #print('debugMode: ', debugMode)

    chat.update(modelName, debugMode, mcp)

    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")
    # logger.info(f"clear_button: {clear_button}")

st.title('🔮 '+ mode)

if clear_button==True:
    chat.initiate()
    cost.cost_data = {}
    cost.visualizations = {}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "images" in message:                
                for url in message["images"]:
                    logger.info(f"url: {url}")

                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)
            st.markdown(message["content"])

display_chat_messages()

def show_references(reference_docs):
    if debugMode == "Enable" and reference_docs:
        with st.expander(f"답변에서 참조한 {len(reference_docs)}개의 문서입니다."):
            for i, doc in enumerate(reference_docs):
                st.markdown(f"**{doc.metadata['name']}**: {doc.page_content}")
                st.markdown("---")

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []        
    
    st.session_state.greetings = False
    chat.clear_chat_history()
    st.rerun()    

# Preview the uploaded image in the sidebar
file_name = ""
    
if clear_button==False and mode == '비용 분석':
    st.subheader("📈 Cost Analysis")

    if not cost.visualizations:
        cost.get_visualiation()

    if 'service_pie' in cost.visualizations:
        st.plotly_chart(cost.visualizations['service_pie'])
    if 'daily_trend' in cost.visualizations:
        st.plotly_chart(cost.visualizations['daily_trend'])
    if 'region_bar' in cost.visualizations:
        st.plotly_chart(cost.visualizations['region_bar'])

    with st.status("thinking...", expanded=True, state="running") as status:
        if not cost.cost_data:
            st.info("비용 데이터를 가져옵니다.")
            cost_data = cost.get_cost_analysis()
            logger.info(f"cost_data: {cost_data}")
            cost.cost_data = cost_data
        else:
            if not cost.insights:        
                st.info("잠시만 기다리세요. 지난 한달간의 사용량을 분석하고 있습니다...")
                insights = cost.generate_cost_insights()
                logger.info(f"insights: {insights}")
                cost.insights = insights
            
            st.markdown(cost.insights)
            st.session_state.messages.append({"role": "assistant", "content": cost.insights})

# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")

    with st.chat_message("assistant"):
        if mode == '일상적인 대화':
            stream = chat.general_conversation(prompt)            
            response = st.write_stream(stream)
            logger.info(f"response: {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})

            chat.save_chat_history(prompt, response)

        elif mode == 'RAG':
            with st.status("running...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_rag_with_knowledge_base(prompt, st)                           
                st.write(response)
                logger.info(f"response: {response}")

                st.session_state.messages.append({"role": "assistant", "content": response})

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 
        
        elif mode == 'Agent':
            sessionState = ""
            chat.references = []
            chat.image_url = []
            response = chat.run_agent(prompt, "Disable", st)

        elif mode == 'Agent (Chat)':
            sessionState = ""
            chat.references = []
            chat.image_url = []
            response = chat.run_agent(prompt, "Enable", st)

        elif mode == "Multi-agent Supervisor (Router)":
            sessionState = ""
            chat.references = []
            chat.image_url = []
            with st.status("thinking...", expanded=True, state="running") as status:
                response, image_url, reference_docs = router.run_router_supervisor(prompt, st)
                st.write(response)
                logger.info(f"response: {response}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "images": image_url if image_url else []
                })
                chat.save_chat_history(prompt, response)       

                show_references(reference_docs)              

        elif mode == "LangGraph Supervisor":
            sessionState = ""
            chat.references = []
            chat.image_url = []
            with st.status("thinking...", expanded=True, state="running") as status:
                response, image_url, reference_docs = supervisor.run_langgraph_supervisor(prompt, st)
                st.write(response)
                logger.info(f"response: {response}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "images": image_url if image_url else []
                })
                chat.save_chat_history(prompt, response)       

                show_references(reference_docs)              

        elif mode == "LangGraph Swarm":
            sessionState = ""
            with st.status("thinking...", expanded=True, state="running") as status:
                response, image_url, reference_docs = swarm.run_langgraph_swarm(prompt, st)
                st.write(response)
                logger.info(f"response: {response}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "images": image_url if image_url else []
                })
                chat.save_chat_history(prompt, response)       

                show_references(reference_docs)              

        elif mode == '번역하기':
            response = chat.translate_text(prompt)
            st.write(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)
        
        elif mode == '비용 분석':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = cost.ask_cost_insights(prompt)
                st.write(response)

                st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            stream = chat.general_conversation(prompt)

            response = st.write_stream(stream)
            logger.info(f"response: {response}")

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)
        