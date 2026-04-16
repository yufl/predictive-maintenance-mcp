from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from src.env_utils import GITHUB_TOKEN, ZHIPU_SECRET_KEY

copilot_class_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com",
)

# Copilot基础套餐支持的模型：openai的gpt-4.1和gpt-4o
# GPT-4.1 代表 OpenAI 在纯文本智能上的巅峰，通过更大上下文、更强指令遵循和编程能力，满足企业级复杂任务需求；
# 而 GPT-4o 则以多模态交互和亲民价格成为大众用户的首选。
# 选择时需权衡：
# 若需处理超长文本或专业编程，GPT-4.1 是更优解；
# 若涉及语音、图像等多模态输入，GPT-4o 仍是不可替代的选择。
copilot_init_llm = init_chat_model(
    model="gpt-4o",
    model_provider="openai",
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com",
)

zhipu_init_llm = init_chat_model(
    model="glm-4.6",
    model_provider="openai",
    api_key=ZHIPU_SECRET_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

# resp = zhipu_init_llm.invoke("介绍一下自己。")
# print(resp)