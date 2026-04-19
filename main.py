import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.react_agent import ReactAgent

# 1. 初始化 FastAPI 应用
app = FastAPI(title="智扫通机器人 API 服务")

# 2. 配置 CORS 跨域 (允许前端跨域访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议设置为具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 初始化 Agent (全局单例)
# 确保 react_agent.py 已按照之前的建议添加了 SqliteSaver 持久化记忆
agent = ReactAgent()

# 4. 定义请求体模型
class ChatRequest(BaseModel):
    user_id: str
    query: str

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口
    """
    def generate():
        # 调用 ReactAgent 中定义的 execute_stream 生成器
        # 它会根据 user_id 自动处理持久化记忆隔离
        for chunk in agent.execute_stream(request.query, user_id=request.user_id):
            # 这里的 chunk 已经是处理好的字符串
            yield chunk

    # 使用 StreamingResponse 返回流式数据
    # media_type 设置为 text/event-stream 或 text/plain
    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    # 启动服务，默认 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)