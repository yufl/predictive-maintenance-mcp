import os
import asyncio
from typing import Annotated, TypedDict, List, Union

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# 基础框架
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 关键组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, filter_messages, trim_messages
from langchain_core.tools import Tool


# 导入你原有的 MCP 实例 (确保路径正确)
# 假设你的入口文件叫 server.py，且位于当前目录下
try:
    from server import mcp
except ImportError:
    # 适配不同的工作路径
    from src.server import mcp


# --- 第一步：工具桥接逻辑 ---

def get_langchain_tools_from_fastmcp(fastmcp_instance):
    """
    将 FastMCP 内部注册的工具自动化转换为 LangChain 工具。
    这样可以保留你在 register_all(mcp) 中定义的所有 ISO 算法逻辑。
    """
    lc_tools = []
    # 检查 FastMCP 内部结构
    # 尝试获取所有已注册的工具对象
    mcp_tools = fastmcp_instance._tool_manager.list_tools()

    for item in mcp_tools:
        # 兼容性处理：根据返回值类型进行解包
        if isinstance(item, tuple):
            # 如果是旧版的 (name, tool_obj)
            name = item[0]
            tool_obj = item[1]
        else:
            # 如果是新版直接返回 ToolDefinition 对象
            tool_obj = item
            name = tool_obj.name

        # 封装为 LangChain Tool
        lc_tools.append(Tool(
            name=name,
            # FastMCP 工具的核心执行函数通常在 .fn 属性中
            func=tool_obj.fn,
            description=tool_obj.description or tool_obj.fn.__doc__
        ))
    return lc_tools

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
llm_model = init_chat_model(
    model="gpt-4o",
    model_provider="openai",
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com",
    temperature=0,  # 诊断任务建议设为 0 以保证严谨性
)

# 定义修剪器
# strategy="last": 保留最后的对话
# token_counter: 使用模型对应的计数器
# max_tokens: 预留空间给 System Message 和工具输出
trimmer = trim_messages(
    max_tokens=4000,
    strategy="last",
    token_counter=llm_model,
    include_system=True, # 确保 System Message 始终被保留
    start_on="human",    # 确保从人类消息开始，避免悬挂的 AI 或 Tool 消息
)

# --- 第二步：定义 LangGraph 状态机 ---

class MaintenanceState(TypedDict):
    # 自动合并历史消息
    messages: Annotated[List[BaseMessage], add_messages]


# 1. 节点：LLM 决策节点
def call_diagnostic_model_bk(state: MaintenanceState):
    """
    使用 Copilot 基础模型 (GPT-4o) 进行推理。
    会自动注入 mcp_server.py 中定义的超长 instructions。
    """
    # 这里的关键是：必须注入你在 mcp 对象中定义的推理规则 (Instructions)
    system_prompt = SystemMessage(content=mcp.instructions)

    # 绑定工具
    tools = get_langchain_tools_from_fastmcp(mcp)
    llm = llm_model.bind_tools(tools)

    chain = llm
    response = chain.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def call_diagnostic_model(state: MaintenanceState):
    # 1. 精简指令：不要直接透传整个 mcp.instructions
    # 提取最核心的政策，过滤掉关于 HTML 结构、文件路径等模型不需要反复阅读的说明
    core_rules = """
    - Use evidence-based inference (ISO 13374).
    - Do NOT make diagnostic claims based solely on statistical parameters.
    - Confirm signal units before ISO 20816 evaluation.
    - Bearing fault must be supported by frequency-domain evidence.
    """

    # system_prompt = SystemMessage(content=core_rules)

    # 1. 组合所有消息
    # 注意：我们将精简后的 Instructions 放在最前面
    system_prompt = SystemMessage(content=mcp.instructions[:2000])  # 截断原始超长指令
    all_messages = [system_prompt] + state["messages"]

    # 2. 执行修剪
    selected_messages = trimmer.invoke(all_messages)

    # 在发送给 Copilot 前，先修剪消息列表
    # trimmed_msgs = trimmer.invoke(state["messages"])

    # 2. 限制历史消息长度 (这是解决 413 的关键)
    # 只取最近的 5-10 轮对话，防止上下文累积导致的 Body 过大
    # trimmed_messages = state["messages"][-10:]

    # 自动保留最近的 5000 tokens 左右的消息，防止超出 8000 的 Body 限制
    # messages = filter_messages(state["messages"], max_tokens=5000)

    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    llm = init_chat_model(
        model="gpt-4o",
        model_provider="openai",
        api_key=GITHUB_TOKEN,
        base_url="https://models.inference.ai.azure.com",
        temperature=0,  # 诊断任务建议设为 0 以保证严谨性
    ).bind_tools(tools)
    # response = llm.invoke([system_prompt] + trimmed_messages)
    # response = llm.invoke([system_prompt] + messages)
    response = llm.invoke([system_prompt] + selected_messages)
    # response = llm.invoke(trimmed_msgs)
    return {"messages": [response]}


# 2. 节点：工具执行节点 (由 LangGraph 预置)
tools = get_langchain_tools_from_fastmcp(mcp)
tool_node = ToolNode(tools)


# --- 第三步：构建工作流图 ---

def should_continue(state: MaintenanceState):
    """判断模型是想调用工具还是直接回复用户"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(MaintenanceState)

# 添加节点
workflow.add_node("agent", call_diagnostic_model)
workflow.add_node("tools", tool_node)

# 设置逻辑连线
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 编译应用
app = workflow.compile()


# --- 第四步：执行与测试接口 ---

async def run_diagnostic_session(user_query: str):
    """
    模拟 Copilot 的交互过程
    """
    print(f"--- 启动工业诊断任务 ---")
    print(f"用户需求: {user_query}\n")

    inputs = {"messages": [HumanMessage(content=user_query)]}

    async for event in app.astream(inputs, stream_mode="values"):
        message = event["messages"][-1]

        # 打印 AI 的思考或工具调用路径
        if hasattr(message, "content") and message.content:
            print(f"AI: {message.content}")

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                print(f"🛠️  执行 MCP 工具: {tc['name']}({tc['args']})")


# --- 入口 ---

if __name__ == "__main__":
    # 模拟一个典型的三轴振动分析场景
    test_query = "查看 1 号机泵的振动信号列表，对其中的异常信号进行轴承故障诊断，并生成 HTML 报告。"

    # 注意：运行此代码需要设置 OPENAI_API_KEY 环境变量
    asyncio.run(run_diagnostic_session(test_query))