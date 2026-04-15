import os
import asyncio
from typing import Annotated, TypedDict, List, Union

# 基础框架
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 关键组件
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, trim_messages, ToolMessage, AIMessage
from langchain_core.tools import Tool
from pydantic import BaseModel

from src.copilot_llm import copilot_init_llm

# 导入你原有的 MCP 实例 (确保路径正确)
# 假设你的入口文件叫 server.py，且位于当前目录下
try:
    from server import mcp
except ImportError:
    # 适配不同的工作路径
    from src.server import mcp


# --- 第一步：工具桥接逻辑 ---
def safe_tool_wrapper(func):
    def wrapper(*args, **kwargs):
        # 核心修复：如果 func 不需要参数但接收到了参数，则忽略位置参数
        try:
            result = func(*args, **kwargs)
        except TypeError:
            # 针对不带参数的函数进行降级调用
            result = func()
        str_result = str(result)
        # 【修改点 3】将截断阈值降至 1500 字符，确保多设备并行时不会爆表
        if len(str_result) > 1500:
            return str_result[:1500] + "\n[Data Truncated for Token Limit]"
        return result
    return wrapper

def wrap_and_trim_messages(left: list, right: list):
    """
    替换默认的 add_messages。
    确保整个 state 里的消息总数不会无限制增长，从源头控制 State 大小。
    """
    combined = left + right
    # 强制只保留最近的 15 条消息（对于分析 3 个设备足够了）
    if len(combined) > 15:
        return combined[-15:]
    return combined

class NoParams(BaseModel):
    """用于无参数工具的空模式"""
    pass

def get_langchain_tools_from_fastmcp(fastmcp_instance):
    """
    将 FastMCP 内部注册的工具自动化转换为 LangChain 工具。
    这样可以保留你在 register_all(mcp) 中定义的所有 ISO 算法逻辑。
    """
    lc_tools = []
    mcp_tools = fastmcp_instance._tool_manager.list_tools()

    for item in mcp_tools:
        # 解包逻辑保持不变
        if isinstance(item, tuple):
            name, tool_obj = item
        else:
            tool_obj = item
            name = tool_obj.name

        lc_tools.append(Tool(
            name=name,
            func=safe_tool_wrapper(tool_obj.fn),
            description=tool_obj.description or tool_obj.fn.__doc__,
            # 如果函数没有参数，通过指定 args_schema 告知 LangChain
            args_schema=NoParams if tool_obj.fn.__code__.co_argcount == 0 else None
        ))
    return lc_tools


# 2. 这里的 max_tokens 必须下调到 2500-3000
# 考虑到 GitHub Models 的 8000 限制包含：Header + System + History + Tool Definitions + Buffer
# 这里的 max_tokens 必须降到极致
# 8000 限制下，建议只给历史留 2500 tokens
local_trimmer = trim_messages(
    max_tokens=2500,
    strategy="last",
    token_counter=copilot_init_llm,
    include_system=True,
    start_on="human",
)

# --- 第二步：定义 LangGraph 状态机 ---

class MaintenanceState(TypedDict):
    # 自动合并历史消息
    messages: Annotated[List[BaseMessage], wrap_and_trim_messages]


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
    llm = copilot_init_llm.bind_tools(tools)

    chain = llm
    response = chain.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def call_diagnostic_model(state: MaintenanceState):
    # 1. 极其激进的清洗：将工具返回内容缩减到 800 字符以内
    processed_messages = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and len(str(msg.content)) > 1000:
            content_str = str(msg.content)
            # 工业诊断只需要关键特征，不需要原始数组
            truncated_content = content_str[:800] + "...[Data Cut]"
            processed_messages.append(ToolMessage(
                content=truncated_content,
                tool_call_id=msg.tool_call_id
            ))
        else:
            processed_messages.append(msg)

    # 彻底放弃 mcp.instructions，改用极简指令
    core_rules = SystemMessage(content="Act as a vibration analyst. Be concise. Use only stats.")

    selected_history = local_trimmer.invoke(processed_messages)
    llm = copilot_init_llm.bind_tools(tools[:5])

    # 只有这里的调用才会产生 413，确保 [core_rules] + selected_history 足够小
    return {"messages": [llm.invoke([core_rules] + selected_history)]}


# 2. 节点：工具执行节点 (由 LangGraph 预置)
tools = get_langchain_tools_from_fastmcp(mcp)
tool_node = ToolNode(tools)


# --- 第三步：构建工作流图 ---

def should_continue(state: MaintenanceState):
    """判断模型是想调用工具还是直接回复用户"""
    last_message = state["messages"][-1]

    # 核心修复：先判断是否为 AIMessage，再检查 tool_calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # 如果是 HumanMessage 或不带工具调用的 AIMessage，则结束
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