import asyncio
import os
from dotenv import load_dotenv

# 导入刚才定义的 bridge_executor 模块
# 假设你已经把上一条回复的代码保存为 bridge_executor.py
from bridge_executor import app, run_diagnostic_session

# 加载环境变量 (包含 OPENAI_API_KEY)
load_dotenv()


async def start_interactive_session():
    """
    启动交互式诊断终端
    """
    print("=" * 50)
    print("工业设备预测性维护 AI 助手 (Powered by LangGraph & FastMCP)")
    print("=" * 50)
    print("输入 'exit' 或 'quit' 退出。")

    while True:
        user_input = input("\n[用户指令] > ")

        if user_input.lower() in ['exit', 'quit']:
            print("正在关闭诊断系统...")
            break

        if not user_input.strip():
            continue

        try:
            # 执行 LangGraph 工作流
            await run_diagnostic_session(user_input)
        except Exception as e:
            print(f"❌ 系统执行错误: {str(e)}")


if __name__ == "__main__":
    # 确保必要的目录存在（复用你 mcp_server 中的逻辑）
    from server import _setup_environment

    _setup_environment()

    # 启动异步循环
    try:
        asyncio.run(start_interactive_session())
    except KeyboardInterrupt:
        pass