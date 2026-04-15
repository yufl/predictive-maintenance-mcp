import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# 1. 加载桥接好的状态机 app 和 原有 MCP 配置
# 这里的 app 是你在上一阶段 compile() 后的结果
from bridge_executor import app
from server import _setup_environment


async def launch_agent():
    # 初始化目录结构 (Reports, Data, Cache 等)
    _setup_environment()
    load_dotenv()

    print("🚀 预测性维护智能体已就绪")
    print("遵循 ISO 13374 诊断标准 | 接入基础模型: GPT-4o\n")

    # 模拟一个典型的工业诊断对话
    user_input = "检查 pump_vibration_001.json 的频谱，并判断是否存在轴承内圈故障？"

    # 2. 启动 LangGraph 异步流
    inputs = {"messages": [HumanMessage(content=user_input)]}

    async for event in app.astream(inputs, stream_mode="values"):
        last_message = event["messages"][-1]

        # 实时打印智能体的决策过程
        if last_message.type == "ai":
            if last_message.tool_calls:
                for tc in last_message.tool_calls:
                    print(f"⚙️  [决策] 正在调用工具: {tc['name']}")
            elif last_message.content:
                print(f"🤖 [诊断结论]:\n{last_message.content}")
        elif last_message.type == "tool":
            print(f"📊 [工具反馈]: 数据处理完成，已获得频谱特征。")


if __name__ == "__main__":
    try:
        asyncio.run(launch_agent())
    except KeyboardInterrupt:
        print("\n智能体已安全停止。")