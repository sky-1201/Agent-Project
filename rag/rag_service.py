"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
(已升级：混合检索 BM25 + Chroma，并引入 BGE-Reranker 重排)
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# 新增：混合检索与重排相关依赖
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
from utils.logger_handler import logger
from utils.config_handler import chroma_conf


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        # 1. 基础服务与模型初始化
        self.vector_store_service = VectorStoreService()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model

        # 2. 构建高级检索链路 (屏蔽底层复杂性)
        self.retriever = self._init_advanced_retriever()

        # 3. 初始化生成的 Chain
        self.chain = self._init_chain()

    def _init_advanced_retriever(self):
        """
        初始化企业级检索链路：Chroma(向量) + BM25(关键字) -> Ensemble(混合) -> BGE Reranker(重排)
        """
        logger.info("[RagSummarizeService] 正在初始化混合检索与重排链路...")

        # --- 步骤 1: 配置基础的向量检索 (召回 Top 10) ---
        vector_retriever = self.vector_store_service.get_retriever()

        # --- 步骤 2: 配置 BM25 关键字检索 (召回 Top 10) ---
        # 从 Chroma 底层获取所有已加载的文本来初始化 BM25
        # 提示：如果知识库极大，企业级应用建议将 BM25 替换为 ElasticsearchRetriever
        try:
            collection_data = self.vector_store_service.vector_store.get()
            all_texts = collection_data.get("documents", [])

            if not all_texts:
                logger.warning("[RagSummarizeService] 向量库为空，BM25 降级，仅使用向量检索。")
                bm25_retriever = vector_retriever  # 降级处理
            else:
                bm25_retriever = BM25Retriever.from_texts(all_texts)
                bm25_retriever.k = chroma_conf["k"]
        except Exception as e:
            logger.error(f"[RagSummarizeService] BM25 初始化失败: {str(e)}。降级为纯向量检索。")
            bm25_retriever = vector_retriever

        # --- 步骤 3: 使用 EnsembleRetriever 混合双路召回 (权重 0.5 : 0.5) ---
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )

        # --- 步骤 4: 引入 CrossEncoder 重排模型 ---
        try:
            # 使用 BAAI 的 bge-reranker-base (轻量且对中文极其友好)
            model_name = "BAAI/bge-reranker-base"
            logger.info(f"[RagSummarizeService] 正在加载重排模型 {model_name}...")

            rerank_model = HuggingFaceCrossEncoder(model_name=model_name)
            # 配置 Reranker，重排混合召回的文档，并截取最终得分最高的 Top 3
            compressor = CrossEncoderReranker(model=rerank_model, top_n=3)

            # --- 步骤 5: 将重排器和混合检索器包装为单一检索器 ---
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever
            )
            logger.info("[RagSummarizeService] 混合检索与重排链路初始化完成。")
            return compression_retriever

        except Exception as e:
            logger.error(f"[RagSummarizeService] 重排模型加载失败: {str(e)}。将退化为不带重排的混合检索。", exc_info=True)
            # 如果重排模型加载失败（比如网络原因没下到权重），降级返回混合检索（需手动限制结果数为3）
            ensemble_retriever.search_kwargs = {"k": 3}
            return ensemble_retriever

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        """
        对内方法：获取检索并重排后的最终文档
        """
        # invoke 会自动走完：向量检索 -> BM25检索 -> 合并去重 -> BGE重排打分 -> 取Top3
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """
        对外暴露的方法：上层调用方完全感知不到底层的多路召回和重排机制
        """
        # 获取 Top 3 精确文档
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        logger.info(f"[RagSummarizeService] 为查询 '{query}' 成功召回并重排得到 {len(context_docs)} 条参考资料")

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()
    # 测试复杂查询，验证专有名词和语义混合理解能力
    print(rag.rag_summarize("机器人工作时受遥控器干扰怎么办？"))