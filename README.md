# MCP Agent 활용하기

MCP(Model Context Protocol)은 생성형 AI application이 외부 데이터를 활용하는 주요한 인터페이스로 빠르게 확산되고 있습니다. 여기에서 MCP를 쉽게 사용해 볼수 있도록 local 환경에서 MCP와 agent를 구성합니다. Cloud 배포등 조금 더 상세한 내용은 [kyopark/mcp](https://github.com/kyopark2014/mcp)을 참조합니다.

## MCP 활용

### MCP Basic

사용자는 자신의 Computer에 설치된 Claude Desktop, Cursor와 같은 AI 도구뿐 아니라 주로 Agent형태로 개발된 어플리케이션을 통해 MCP 서버에 연결할 수 있습니다. MCP server는 MCP client의 요청에 자신이 할수 있는 기능을 capability로 제공하고 client의 요청을 수행합니다. MCP server는 local computer의 파일이나 데이터베이스를 조회할 수 있을뿐 아니라 인터넷에 있는 외부 서버의 API를 이용해 필요한 정보를 조회할 수 있습니다. MCP Client는 Server와 JSON-RPC 2.0 프로토콜을 이용해 연결되는데, stdio나 SSE (Server-Sent Events)을 선택하여, Host의 요청을 MCP에 전달할 수 있고, 응답을 받아서 활용할 수 있습니다.  

<img src="https://github.com/user-attachments/assets/36d2d24c-865b-4b71-b708-5c611ab7785e" width="750">


MCP의 주요 요소의 정의와 동작은 아래와 같습니다.

- MCP Hosts: MCP 프로토콜을 통해 데이터에 접근하는 프로그램/AI 도구로서 Claude Desktop, Cursor, User Agent Application이 해당됩니다.
- MCP Clients: MCP Server와 1:1로 연결을 수행하는 Client로서 MCP Server와 stdio 또는 SSE 방식으로 연결할 수 있습니다.
- MCP Servers: 표준화된 MCP를 통해 Client에 Tool의 Capability를 알려주는 경량 프로그램으로 Local Computer의 파일이나 데이터베이스를 조회할 수 있고, 외부 API를 이용해 정보를 조회할 수 있습니다.
- Local data sources: MCP 서버가 접근할 수 있는 데이터베이스와 로컬 데이터
- Remote services: API를 통해 접근 가능한 외부 시스템

MCP를 사용하면 아래와 같은 장점이 있습니다.

- 표준화된 방식으로 다양한 데이터 소스에 접근 가능합니다.
- 애플리케이션 코드 변경 없이 MCP 서버 업데이트를 통한 새로운 기능 추가할 수 있습니다.
- 조직 전반에 걸쳐 AI 지원 및 확장이 용이합니다.


[MCP Server Components](https://www.philschmid.de/mcp-introduction)에는 아래와 같은 항목이 있습니다. 

- Tools (Model-controlled): LLM이 특정 작업을 수행하기 위해 호출할 수 있는 기능(도구)으로서, API와 같이 특정한 action을 수행합니다. 

```python
tools = await session.list_tools()
```

- Resources (Application-controlled): 생성형 AI 어플리케이션이 접근 할 수 있는 데이터 소스입니다. 복잡한 계산(significant computation)이나 부작용(side effect)없이 데이터를 가져올 수 있습니다. 

```python
resources = await session.list_resources()
```

- Prompts (User-controlled): tool나 resource를 사용할때에 이용하는 사전 정의된 템플렛으로서 추론(inference)전에 선택할 수 있습니다.

```python
prompts = await session.list_prompts()
```
  
### LangChain MCP Adapter

[LangChain MCP Adapter](https://github.com/langchain-ai/langchain-mcp-adapters)는 MCP를 LangGraph agent와 함께 사용할 수 있게 해주는 경량의 랩퍼(lightweight wrapper)로서 MIT 기반의 오픈소스입니다. MCP Adapter의 주된 역할은 MCP server를 위한 tool들을 정의하고, MCP client에서 tools의 정보를 조회하고 LangGraph의 tool node로 정의하여 활용할 수 있도록 도와줍니다. 


#### MCP Server

RAG 검색을 위한 MCP server는 아래와 같이 정의할 수 있습니다. Server의 transport를 "stdio"로 지정하면 server를 지속 실행시키지 않더라도, client가 server의 python code를 직접 실행할 수 있어서 편리합니다. 

```python
from mcp.server.fastmcp import FastMCP 

mcp = FastMCP(
    name = "Search",
    instructions=(
        "You are a helpful assistant. "
        "You can search the documentation for the user's question and provide the answer."
    ),
) 

@mcp.tool()
def search(keyword: str) -> str:
    "search keyword"

    return retrieve_knowledge_base(keyword)

if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")
```

Server는 요청이 들어오면, retrieve_knowledge_base()로 RAG 검색을 수행합니다. Server의 python code는 경량(lightweight)이어야 하므로, 아래와 같이 lambda를 trigger하는 방식으로 구성하였습니다. Lambda에서는 retrieve, grade, generation의 동작을 수행합니다. 아래와 같이 "model_name"을 지정할 수 있고, 필요에 따라서는 "grading"을 선택적으로 사용할 수 있습니다. 또한 병렬처리로 속도를 빠르게 하고 싶은 경우에은 "multi_region"을 "Enable"로 설정합니다. 상세한 코드는 [lambda-rag](./lambda-rag/lambda_function.py)를 참조합니다. 

```python
def retrieve_knowledge_base(query):
    lambda_client = boto3.client(
        service_name='lambda',
        region_name=bedrock_region
    )
    functionName = f"lambda-rag-for-{projectName}"
    payload = {
        'function': 'search_rag',
        'knowledge_base_name': knowledge_base_name,
        'keyword': query,
        'top_k': numberOfDocs,
        'grading': "Enable",
        'model_name': model_name,
        'multi_region': multi_region
    }
    output = lambda_client.invoke(
        FunctionName=functionName,
        Payload=json.dumps(payload),
    )
    payload = json.load(output['Payload'])
    return payload['response'], []
```

#### MCP Client

MCP client이 하나의 MCP server만 볼 경우에는 아래와 같이 stdio_client와 StdioServerParameters를 이용해 구현할 수 있습니다. MCP server에 대한 정보는 config.json에서 읽어오거나 streamlit에서 사용자가 입력한 정보를 사용할 수 있습니다. load_mcp_server_parameters()에서는 mcp_json을 읽어와서 [StdioServerParameters](https://github.com/langchain-ai/langchain-mcp-adapters)을 구성합니다. config.json의 MCP server에 대한 정보는 AWS CDK로 배포후 생성되는 output에서 가져옵니다.

```python
from mcp import ClientSession, StdioServerParameters

def load_mcp_server_parameters():
    mcp_json = json.loads(mcp_config)
    mcpServers = mcp_json.get("mcpServers")

    command = ""
    args = []
    if mcpServers is not None:
        for server in mcpServers:
            config = mcpServers.get(server)
            if "command" in config:
                command = config["command"]
            if "args" in config:
                args = config["args"]
            break

    return StdioServerParameters(
        command=command,
        args=args
    )
```

아래와 같이 MCP server에 대한 정보로 stdio_client를 구성합니다. 이때 tools에 대한 정보를 load_mcp_tools로 가져옵니다. Agent에서는 tool 정보를 bind하고 ainvoke를 이용해 요청된 동작을 수행합니다. 

```python
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

async def mcp_rag_agent_single(query, st):
    server_params = load_mcp_server_parameters()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            with st.status("thinking...", expanded=True, state="running") as status:       
                agent = create_agent(tools)
                agent_response = await agent.ainvoke({"messages": query})                

                result = agent_response["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result
            })            
            return result
```

MCP client는 아래와 같이 실행합니다. 비동기적으로 실행하기 위해서 asyncio를 이용하였습니다.  이후 사용자가 UI에서 MCP Config를 업데이트하면 정보를 업데이트 할 수 있습니다. 

```python
asyncio.run(mcp_rag_agent_single(query, st))
```

서버 정보가 여럿인 경우에 [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)에서 제공하는 MultiServerMCPClient을 이용합니다. 먼저, 아래와 같이 서버 정보를 가져옵니다. 

```python
def load_multiple_mcp_server_parameters():
    mcp_json = json.loads(mcp_config)
    mcpServers = mcp_json.get("mcpServers")

    server_info = {}
    if mcpServers is not None:
        command = ""
        args = []
        for server in mcpServers:
            config = mcpServers.get(server)
            if "command" in config:
                command = config["command"]
            if "args" in config:
                args = config["args"]

            server_info[server] = {
                "command": command,
                "args": args,
                "transport": "stdio"
            }
    return server_info
```

이후 아래와 같이 MCP server정보와 MultiServerMCPClient로 client를 정의합니다. MCP server로 부터 가져온 tool 정보는 client.get_tools()로 가져와서 agent를 생성할 때에 사용합니다. Single MCP server와 마찬가지로 ainvoke로 실행하여 결과를 얻을 수 있습니다. 

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
asyncio.run(mcp_rag_agent_multiple(query, st))

async def mcp_rag_agent_multiple(query, st):
    server_params = load_multiple_mcp_server_parameters()
    async with  MultiServerMCPClient(server_params) as client:
        with st.status("thinking...", expanded=True, state="running") as status:                       
            tools = client.get_tools()
            agent = create_agent(tools)
            response = await agent.ainvoke({"messages": query})
            result = response["messages"][-1].content

        st.markdown(result)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result
        })
    return result
```

여기서는 customize가 용이하도록 agent를 정의하였습니다.

```python
def create_agent(tools):
    tool_node = ToolNode(tools)

    chatModel = get_chat(extended_thinking="Disable")
    model = chatModel.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def call_model(state: State, config):
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
        return {"messages": [response]}

    def should_continue(state: State) -> Literal["continue", "end"]:
        messages = state["messages"]    
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"        
        else:
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
    
    return buildChatAgent()
```

## MCP Servers의 활용

[Model Context Protocol servers](https://github.com/modelcontextprotocol/servers)에서도 아래와 같은 서버들에 대한 정보를 제공하고 있습니다.

- [Perplexity Ask MCP Server](https://github.com/ppl-ai/modelcontextprotocol)
- [Riza MCP Server](https://github.com/riza-io/riza-mcp)
- [Tavily MCP Server](https://github.com/tavily-ai/tavily-mcp)

[Smithery](https://smithery.ai/)에서 MCP server를 찾아보고 필요한 서버를 찾으면 접속할 수 있는 MCP 서버 정보를 JSON 형태로 조회할 수 있습니다. 

<img src="https://github.com/user-attachments/assets/62e534ee-88bd-4f9f-a4ff-129522fd834f" width="500">

[Smithery - Google Search Server](https://smithery.ai/server/@gradusnikov/google-search-mcp-server)에서 확인한 구글 검색용 MCP 서버 정보는 아래와 같습니다. 검색엔진 ID와 API Key를 필요로 합니다. 

```java
{
  "mcpServers": {
    "google-search-mcp-server": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@gradusnikov/google-search-mcp-server",
        "--config",
        "{\"googleCseId\":\"b5cd8c527fbd64b72\",\"googleApiKey\":\"AIzbSyDQlYpck8-9TbBSuxoew1luOGVB6unRPNk\"}"
      ]
    }
  }
}
```

아래와 같이 json 형식의 서버정보를 업데이트 할 수 있습니다. 아래에서는 [mcp-server.py](./application/mcp-server.py)에서 정의한 search를 이용하고 있습니다.

```java
{
  "mcpServers": {
    "search": {
      "command": "python",
      "args": [
        "application/mcp-server.py"
      ]
    }
  }
}
```

### 실행하기

venv로 환경을 구성하면 편리합니다. 아래와 같이 환경을 설정합니다.

```text
python -m venv venv
source venv/bin/activate
```

이후 다운로드 받은 github 폴더로 이동한 후에 아래와 같이 필요한 패키지를 추가로 설치 합니다.

```text
python -m pip install -r requirements.txt
```

아래와 같은 명령어로 streamlit을 실행합니다. 

```text
streamlit run application/app.py
```

### 실행 준비

Diagram을 그리기 위해서는 [Graphviz](https://www.graphviz.org/download/)를 따라서 graphviz를 설치합니다. Mac에서는 아래 명령어를 사용합니다.

```text
brew install graphviz
```

## 실행 결과

"서울에서 부산을 거쳐서 제주에 가려고 합니다. 가는 동안의 날씨와 맛집은?"와 같이 복잡 질문을 수행하면 아래와 같이 여러가지 tool들을 이용하여 정보를 수집합니다. 

![image](https://github.com/user-attachments/assets/1472c28b-e805-422c-b3ee-5a1246acf0cf)

결과는 아래와 같이 복잡 질문에 대해 답변이 가능합니다.

![image](https://github.com/user-attachments/assets/c8909669-c5b9-4b5a-a498-eb7f1b4db37f)

## Reference 

[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

[LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
