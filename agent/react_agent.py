import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver  # 引入 SQLite 检查点
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch

# 注意：这里的 create_agent 假设是你项目中封装好的 LangGraph 构建函数
from langchain.agents import create_agent


class ReactAgent:
    def __init__(self):
        # 1. 初始化 SQLite 连接并创建持久化存储实例
        # check_same_thread=False 允许 Streamlit 等多线程环境安全访问
        self.conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
        self.memory = SqliteSaver(self.conn)

        # 2. 在创建 Agent 时注入 checkpointer
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                    fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            checkpointer=self.memory,  # 关键：配置持久化记忆模块
        )

    def execute_stream(self, query: str, user_id: str = "default_user"):
        """
        修改后的执行方法，支持传入 user_id 以隔离不同用户的状态
        """
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 3. 配置 thread_id，LangGraph 会根据此 ID 自动从数据库加载/保存对应的历史记录
        config = {"configurable": {"thread_id": user_id}}

        # 在 stream 调用中传入 config
        # context 用于触发提示词切换逻辑
        for chunk in self.agent.stream(
                input_dict,
                config=config,
                stream_mode="values",
                context={"report": False}
        ):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    # 测试代码
    agent = ReactAgent()
    # 模拟用户 1001 的连续对话
    print("--- 第一次对话 ---")
    for chunk in agent.execute_stream("你好，我是小明", user_id="1001"):
        print(chunk, end="")

    print("\n--- 第二次对话（测试记忆） ---")
    for chunk in agent.execute_stream("我刚才说我叫什么？", user_id="1001"):
        print(chunk, end="")