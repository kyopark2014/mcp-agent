import traceback
import boto3
import os
import json
import re
import uuid
import time
import base64
import info 
import PyPDF2
import csv
import asyncio

from io import BytesIO
from PIL import Image
from langchain_aws import ChatBedrock
from botocore.config import Config
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import tool
from langchain.docstore.document import Document
from tavily import TavilyClient  
from langchain_community.tools.tavily_search import TavilySearchResults
from urllib import parse
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import ToolNode
from typing import Literal
from langgraph.graph import START, END, StateGraph
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from multiprocessing import Process, Pipe

import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("chat")

userId = uuid.uuid4().hex
map_chain = dict() 

checkpointers = dict() 
memorystores = dict() 

checkpointer = MemorySaver()
memorystore = InMemoryStore()

checkpointers[userId] = checkpointer
memorystores[userId] = memorystore

reasoning_mode = 'Disable'
debug_messages = []  # List to store debug messages

def get_debug_messages():
    global debug_messages
    messages = debug_messages.copy()
    debug_messages = []  # Clear messages after returning
    return messages

def push_debug_messages(type, contents):
    global debug_messages
    debug_messages.append({
        type: contents
    })

def status_messages(message):
    # type of message
    if isinstance(message, AIMessage):
        logger.info(f"status_messages (AIMessage): {message}")
    elif isinstance(message, ToolMessage):
        logger.info(f"status_messages (ToolMessage): {message}")
    elif isinstance(message, HumanMessage):
        logger.info(f"status_messages (HumanMessage): {message}")

    if isinstance(message, AIMessage):
        if message.content:
            logger.info(f"content: {message.content}")
            content = message.content
            if len(content) > 500:
                content = content[:500] + "..."       
            push_debug_messages("text", content)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            logger.info(f"Tool name: {message.tool_calls[0]['name']}")
                
            if 'args' in message.tool_calls[0]:
                logger.info(f"Tool args: {message.tool_calls[0]['args']}")
                    
                args = message.tool_calls[0]['args']
                if 'code' in args:
                    logger.info(f"code: {args['code']}")
                    push_debug_messages("text", args['code'])
                elif message.tool_calls[0]['args']:
                    status = f"Tool name: {message.tool_calls[0]['name']}  \nTool args: {message.tool_calls[0]['args']}"
                    logger.info(f"status: {status}")
                    push_debug_messages("text", status)

    elif isinstance(message, ToolMessage):
        if message.name:
            logger.info(f"Tool name: {message.name}")
            
            if message.content:                
                content = message.content
                if len(content) > 500:
                    content = content[:500] + "..."
                logger.info(f"Tool result: {content}")                
                status = f"Tool name: {message.name}  \nTool result: {content}"
            else:
                status = f"Tool name: {message.name}"

            logger.info(f"status: {status}")
            push_debug_messages("text", status)

def initiate():
    global userId
    global memory_chain, checkpointers, memorystores, checkpointer, memorystore

    userId = uuid.uuid4().hex
    logger.info(f"userId: {userId}")

    if userId in map_chain:  
            # print('memory exist. reuse it!')
            memory_chain = map_chain[userId]

            checkpointer = checkpointers[userId]
            memorystore = memorystores[userId]
    else: 
        # print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

        checkpointer = MemorySaver()
        memorystore = InMemoryStore()

        checkpointers[userId] = checkpointer
        memorystores[userId] = memorystore

initiate()

bedrock_region = "us-west-2"
projectName = "mcp-agent"

MSG_LENGTH = 100    

model_name = "Claude 3.5 Sonnet"
model_type = "claude"
debug_mode = "Enable"

client = boto3.client(
    service_name='bedrock-agent',
    region_name=bedrock_region
)  

mcp_json = ""
def update(modelName, debugMode, mcp):    
    global model_name, model_id, model_type, debug_mode, multi_region
    global models, mcp_json
    
    if model_name != modelName:
        model_name = modelName
        logger.info(f"model_name: {model_name}")
        
        models = info.get_model_info(model_name)
        model_id = models[0]["model_id"]
        model_type = models[0]["model_type"]
                                
    if debug_mode != debugMode:
        debug_mode = debugMode
        logger.info(f"debug_mode: {debug_mode}")

    mcp_json = mcp
    logger.info(f"mcp_json: {mcp_json}")

def clear_chat_history():
    memory_chain = []
    map_chain[userId] = memory_chain

def save_chat_history(text, msg):
    memory_chain.chat_memory.add_user_message(text)
    memory_chain.chat_memory.add_ai_message(msg) 

selected_chat = 0
def get_llm():
    if model_name == "Claude 3.7 Sonnet":
        modelId = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    elif model_name == "Claude 3.5 Sonnet":
        modelId = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    elif model_name == "Claude 3.0 Sonnet":
        modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    elif model_name == "Claude 3.5 Haiku":
        modelId = "anthropic.claude-3-5-haiku-20241022-v1:0"

    STOP_SEQUENCE = "\n\nHuman:" 
    maxOutputTokens = 4096 
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [STOP_SEQUENCE]
    }

    llm = ChatBedrock(   
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
        region_name=bedrock_region
    )    
    
    return llm

def translate_text(text):
    llm = get_llm()

    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | llm    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        logger.info(f"translated text: {msg}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")      
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag
        
def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags." 
        "Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content

    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")     
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

####################### LangChain #######################
# General Conversation
#########################################################
def general_conversation(query):
    llm = get_llm()

    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )
    
    human = "Question: {input}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        MessagesPlaceholder(variable_name="history"), 
        ("human", human)
    ])
                
    history = memory_chain.load_memory_variables({})["chat_history"]

    chain = prompt | llm | StrOutputParser()
    try: 
        stream = chain.stream(
            {
                "history": history,
                "input": query,
            }
        )  
        logger.info(f"stream: {stream}")
            
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")      
        raise Exception ("Not able to request to LLM: "+err_msg)
        
    return stream

fileId = uuid.uuid4().hex
 
####################### Agent #######################
# Agent 
#####################################################
def create_agent(tools, historyMode):
    tool_node = ToolNode(tools)

    chatModel = get_llm()
    model = chatModel.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        image_url: list

    def call_model(state: State, config):
        logger.info(f"###### call_model ######")
        logger.info(f"state: {state['messages']}")

        last_message = state['messages'][-1].content
        logger.info(f"last message: {last_message}")
        
        # get image_url from state
        image_url = state['image_url'] if 'image_url' in state else []
        if isinstance(last_message, str) and (last_message.strip().startswith('{') or last_message.strip().startswith('[')):
            try:                 
                tool_result = json.loads(last_message)
                if "path" in tool_result:
                    logger.info(f"path: {tool_result['path']}")

                    path = tool_result['path']
                    if isinstance(path, list):
                        for p in path:
                            logger.info(f"image: {p}")
                            if p.startswith('http') or p.startswith('https'):
                                image_url.append(p)
                    else:
                        logger.info(f"image: {path}")
                        if path.startswith('http') or path.startswith('https'):
                            image_url.append(path)
            except json.JSONDecodeError:
                tool_result = last_message
        if image_url:
            logger.info(f"image_url: {image_url}")

        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
            "한국어로 답변하세요."
        )

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            chain = prompt | model
                
            response = chain.invoke(state["messages"])
            # logger.info(f"call_model response: {response}")
            logger.info(f"call_model: {response.content}")

        except Exception:
            response = AIMessage(content="답변을 찾지 못하였습니다.")

            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")
            # raise Exception ("Not able to request to LLM")

        return {"messages": [response], "image_url": image_url}

    def should_continue(state: State) -> Literal["continue", "end"]:
        logger.info(f"###### should_continue ######")

        messages = state["messages"]    
        last_message = messages[-1]
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            tool_name = last_message.tool_calls[-1]['name']
            logger.info(f"--- CONTINUE: {tool_name} ---")

            if debug_mode == "Enable":
                status_messages(last_message)

            return "continue"
        else:
            logger.info(f"--- END ---")
            return "end"

    def buildChatAgent():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")

        return workflow.compile() 
    
    def buildChatAgentWithHistory():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")
    
        return workflow.compile(
            checkpointer=checkpointer,
            store=memorystore
        )
    
    # workflow 
    if historyMode == "Enable":
        app = buildChatAgentWithHistory()
        config = {
            "recursion_limit": 50,
            "configurable": {"thread_id": userId}
        }
    else:
        app = buildChatAgent()
        config = {
            "recursion_limit": 50
        }

    return app, config

def load_mcp_server_parameters():
    logger.info(f"mcp_json: {mcp_json}")

    mcpServers = mcp_json.get("mcpServers")
    logger.info(f"mcpServers: {mcpServers}")

    command = ""
    args = []
    if mcpServers is not None:
        for server in mcpServers:
            logger.info(f"server: {server}")

            config = mcpServers.get(server)
            logger.info(f"config: {config}")

            if "command" in config:
                command = config["command"]
            if "args" in config:
                args = config["args"]
            if "env" in config:
                env = config["env"]

            break

    return StdioServerParameters(
        command=command,
        args=args,
        env=env
    )

def load_multiple_mcp_server_parameters():
    logger.info(f"mcp_json: {mcp_json}")

    mcpServers = mcp_json.get("mcpServers")
    logger.info(f"mcpServers: {mcpServers}")
  
    server_info = {}
    if mcpServers is not None:
        command = ""
        args = []
        for server in mcpServers:
            logger.info(f"server: {server}")

            config = mcpServers.get(server)
            logger.info(f"config: {config}")

            if "command" in config:
                command = config["command"]
            if "args" in config:
                args = config["args"]
            if "env" in config:
                env = config["env"]

                server_info[server] = {
                    "command": command,
                    "args": args,
                    "env": env,
                    "transport": "stdio"
                }
            else:
                server_info[server] = {
                    "command": command,
                    "args": args,
                    "transport": "stdio"
                }
    logger.info(f"server_info: {server_info}")

    return server_info

def tool_info(tools, st):
    tool_info = ""
    tool_list = []
    st.info("Tool 정보를 가져옵니다.")
    for tool in tools:
        tool_info += f"name: {tool.name}\n"    
        if hasattr(tool, 'description'):
            tool_info += f"description: {tool.description}\n"
        tool_info += f"args_schema: {tool.args_schema}\n\n"
        tool_list.append(tool.name)
    st.info(f"Tools: {tool_list}")

async def mcp_rag_agent(query, historyMode, st):
    server_params = load_multiple_mcp_server_parameters()
    logger.info(f"server_params: {server_params}")

    async with MultiServerMCPClient(server_params) as client:
        ref = ""
        with st.status("thinking...", expanded=True, state="running") as status:
            tools = client.get_tools()
            if debug_mode == "Enable":
                tool_info(tools, st)
                logger.info(f"tools: {tools}")

            agent, config = create_agent(tools, historyMode)

            try:
                response = await agent.ainvoke({"messages": query}, config)
                logger.info(f"response: {response}")

                result = response["messages"][-1].content
                # logger.info(f"result: {result}")

                debug_msgs = get_debug_messages()
                for msg in debug_msgs:
                    logger.info(f"debug_msg: {msg}")
                    if "image" in msg:
                        st.image(msg["image"])
                    elif "text" in msg:
                        st.info(msg["text"])

                image_url = response["image_url"] if "image_url" in response else []
                logger.info(f"image_url: {image_url}")

                for image in image_url:
                    st.image(image)

                st.markdown(result)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result,
                    "images": image_url if image_url else []
                })

                return result
            except Exception as e:
                logger.error(f"Error during agent invocation: {str(e)}")
                raise Exception(f"Agent invocation failed: {str(e)}")

def run_agent(query, historyMode, st):
    result = asyncio.run(mcp_rag_agent(query, historyMode, st))

    logger.info(f"result: {result}")
    
    return result
