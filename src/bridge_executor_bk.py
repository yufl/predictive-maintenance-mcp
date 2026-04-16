import functools
import inspect
import os
import asyncio
from typing import Annotated, TypedDict, List, Union

import tiktoken

# 基础框架
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 关键组件
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, trim_messages, ToolMessage, AIMessage
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel

from src.copilot_llm import copilot_init_llm, zhipu_init_llm

# 导入你原有的 MCP 实例 (确保路径正确)
# 假设你的入口文件叫 server.py，且位于当前目录下
try:
    from server import mcp
except ImportError:
    # 适配不同的工作路径
    from src.server import mcp


# --- 第一步：工具桥接逻辑 ---
def safe_tool_wrapper(func):
    """
    更稳健的包装器：
    1. 自动处理 ctx 参数
    2. 物理截断返回内容 (解决 8000 Token 限制)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 检查函数签名
        sig = inspect.signature(func)
        params = sig.parameters

        # 如果函数需要 ctx 但 kwargs 里没有，自动补一个 None
        if 'ctx' in params and 'ctx' not in kwargs and len(args) == 0:
            kwargs['ctx'] = None

        # 执行原始函数
        result = func(*args, **kwargs)

        # 结果截断：针对 8000 tokens 限制，强制将工具输出控制在 800 字符以内
        # 这样即使分析 3 个设备，ToolMessage 总量也不会超标
        str_result = str(result)
        if len(str_result) > 800:
            return str_result[:800] + "\n[数据过长已截断，请基于特征值推理]"
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

def get_langchain_tools_from_fastmcp(fastmcp_instance):
    """
    将 FastMCP 内部注册的工具自动化转换为 LangChain 工具。
    这样可以保留你在 register_all(mcp) 中定义的所有 ISO 算法逻辑。
    """
    lc_tools = []
    mcp_tools = fastmcp_instance._tool_manager.list_tools()

    for item in mcp_tools:
        # 解包 FastMCP 工具对象
        tool_obj = item[1] if isinstance(item, tuple) else item

        # 使用 StructuredTool.from_function 自动处理参数架构
        # 它能完美识别无参数函数 (如 list_signals) 并防止 Argument 冲突
        lc_tools.append(StructuredTool.from_function(
            func=safe_tool_wrapper(tool_obj.fn),
            name=tool_obj.name,
            description=tool_obj.description or tool_obj.fn.__doc__
        ))
    return lc_tools


# --- 第一步：定义自定义计数逻辑 ---
def custom_token_counter(messages: List[BaseMessage]) -> int:
    """
    计算消息列表的总 token 数。
    由于 GLM-4.6 没有内置支持，我们使用 cl100k_base 编码进行估算。
    """
    # 获取编码器 (gpt-4 使用的编码器与大多数现代大模型相似)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # 如果无法获取 tiktoken，退而求其次使用字符长度估算 (1 token ≈ 1.5 - 2 汉字)
        encoding = None

    total_tokens = 0
    for msg in messages:
        # 基础开销：每条消息约 4 tokens (role, name 等字段)
        total_tokens += 4

        # 处理消息文本
        content = msg.content
        if isinstance(content, str):
            if encoding:
                total_tokens += len(encoding.encode(content))
            else:
                total_tokens += len(content) // 1.5  # 简单估算

        # 如果是 AIMessage 且包含工具调用，也需要计入参数的 token
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                total_tokens += len(str(tc)) // 2  # 粗略估算工具调用参数

    return int(total_tokens)

# 2. 针对 8000 限制，将修剪器的阈值设为 2500 tokens
# 给模型回复和工具定义留出足够空间
local_trimmer = trim_messages(
    max_tokens=2500,
    strategy="last",
    token_counter=custom_token_counter,
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
    llm = zhipu_init_llm.bind_tools(tools)

    chain = llm
    response = chain.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


# def call_diagnostic_model(state: MaintenanceState):
#     # 1. 极其激进的清洗：将工具返回内容缩减到 800 字符以内
#     processed_messages = []
#     for msg in state["messages"]:
#         if isinstance(msg, ToolMessage) and len(str(msg.content)) > 1000:
#             content_str = str(msg.content)
#             # 工业诊断只需要关键特征，不需要原始数组
#             truncated_content = content_str[:800] + "...[Data Cut]"
#             processed_messages.append(ToolMessage(
#                 content=truncated_content,
#                 tool_call_id=msg.tool_call_id
#             ))
#         else:
#             processed_messages.append(msg)
#
#     # 彻底放弃 mcp.instructions，改用极简指令
#     # core_rules = SystemMessage(content="Act as a vibration analyst. Be concise. Use only stats.")
#
#     # 1. 定义极其精简的系统指令
#     core_rules = SystemMessage(content="""你是工业诊断专家。
#         1. 仅基于统计特征(RMS/峰值)进行 ISO 20816 评价。
#         2. 如果工具输出被截断，请利用现有数据给出结论。
#         3. 报告需简洁，严禁输出原始波形。""")
#
#     selected_history = local_trimmer.invoke(processed_messages)
#
#     # 3. 绑定工具并调用
#     # llm = zhipu_init_llm.bind_tools(tools[:5])
#     llm = zhipu_init_llm.bind_tools(tools)
#     response = llm.invoke([core_rules] + selected_history)
#
#     return {"messages": [response]}


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

    raw_history = local_trimmer.invoke(processed_messages)

    # 【新增修复逻辑】
    clean_history = []
    for i, msg in enumerate(raw_history):
        # 1. 跳过内容为空的消息
        if not msg.content and (not isinstance(msg, AIMessage) or not msg.tool_calls):
            continue

        # 2. 确保 ToolMessage 之前一定有一个带 tool_calls 的 AIMessage
        if isinstance(msg, ToolMessage):
            if i == 0 or not (isinstance(raw_history[i - 1], AIMessage) and raw_history[i - 1].tool_calls):
                # 如果 ToolMessage 孤立了，为了防止报错，将其转为 HumanMessage 说明
                clean_history.append(HumanMessage(content=f"[工具返回结果]: {msg.content}"))
                continue

        clean_history.append(msg)

    # 3. 确保第一条消息不是 ToolMessage
    while clean_history and isinstance(clean_history[0], ToolMessage):
        clean_history.pop(0)

    # 4. 再次检查是否为空
    if not clean_history:
        clean_history = [HumanMessage(content="继续分析。")]

    # 定义系统指令
    core_rules = SystemMessage(content="你是工业诊断专家，请提供精简的 ISO 13374 分析报告。")

    # 执行调用
    llm = zhipu_init_llm.bind_tools(tools)
    try:
        # 组装：System + 处理后的历史
        response = llm.invoke([core_rules] + clean_history)
        return {"messages": [response]}
    except Exception as e:
        # 如果还是报错，尝试最激进的策略：只发最后一条 HumanMessage
        print(f"警告：模型调用失败，尝试降级。错误: {e}")
        last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        return {"messages": [llm.invoke([core_rules, last_user_msg])]}

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
workflow.add_node("agent", call_diagnostic_model_bk)
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