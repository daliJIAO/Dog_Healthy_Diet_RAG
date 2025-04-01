import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 初始化对话历史
conversation_history = []
DEFAULT_HISTORY_LENGTH = 5

def create_qa_chain(model_name, embedding_model_name, k, nprobe, ef_search):
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)

    # 设置检索器
    retriever = db.as_retriever(
        search_kwargs={"k": int(k), "nprobe": int(nprobe), "efSearch": int(ef_search)}
    )
    llm = Ollama(model=model_name)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

def answer_query(query, history_length, embedding_model_name, base_model_name, k, nprobe, ef_search):
    global conversation_history

    # 限制历史长度
    if len(conversation_history) > history_length * 2:
        conversation_history = conversation_history[-history_length * 2:]

    conversation_history.append(f"用户: {query}")
    qa_chain = create_qa_chain(base_model_name, embedding_model_name, k, nprobe, ef_search)
    full_context = "\n".join(conversation_history)
    result = qa_chain.invoke(full_context)

    conversation_history.append(f"模型: {result['result']}")

    sources = [
        f"{i + 1}. 文件名: {doc.metadata.get('source', '未知')} | 内容片段: {doc.page_content[:60]}..."
        for i, doc in enumerate(result["source_documents"])
    ]
    return result['result'], sources, ""

def clear_history():
    global conversation_history
    conversation_history.clear()
    return "", "", "✅ 对话历史已清除"

def gradio_interface():
    model_options = ["deepseek-r1:7b", "llama3"]
    embedding_options = ["all-MiniLM-L6-v2", "text2vec-base-chinese"]

    with gr.Blocks() as demo:
        gr.Markdown("## 🧪 RAG 问答系统 · 检索参数调节面板")

        with gr.Column():
            query_input = gr.Textbox(label="🧠 请输入你的问题", placeholder="例如：如何为萨摩耶定制营养食谱？")
            result_output = gr.Textbox(label="🤖 模型回答", lines=4)
            source_output = gr.JSON(label="📚 相关文档")
            status_output = gr.Textbox(label="⚠️ 系统状态/提示", interactive=False)

        with gr.Row():
            history_input = gr.Number(label="🕗 限制对话历史轮数", value=DEFAULT_HISTORY_LENGTH, precision=0,
                                      info="建议设置为5，可控制上下文长度")
            embedding_dropdown = gr.Dropdown(choices=embedding_options, label="📌 选择词嵌入模型",
                                             value="all-MiniLM-L6-v2", info="需与 ingest.py 中模型一致")
            base_model_dropdown = gr.Dropdown(choices=model_options, label="📦 选择基底大模型", value="deepseek-r1:7b")

        with gr.Row():
            k_slider = gr.Slider(1, 10, value=3, step=1, label="🔍 k（返回前k个文档）",
                                 info="越大结果越全，响应时间也更久")
            nprobe_slider = gr.Slider(1, 50, value=10, step=1, label="⚙️ nprobe（搜索簇数量）",
                                      info="适用于 IVF 索引，提高召回率")
            ef_slider = gr.Slider(10, 100, value=40, step=1, label="📈 efSearch（HNSW 搜索广度）",
                                  info="适用于 HNSW 索引，提高准确率")

        with gr.Row():
            submit_btn = gr.Button("🚀 提交提问")
            clear_btn = gr.Button("🧹 清除历史")

        # 事件绑定
        submit_btn.click(
            fn=answer_query,
            inputs=[query_input, history_input, embedding_dropdown, base_model_dropdown,
                    k_slider, nprobe_slider, ef_slider],
            outputs=[result_output, source_output, status_output]
        )
        clear_btn.click(fn=clear_history, inputs=[], outputs=[result_output, source_output, status_output])

    return demo

# 启动 Gradio 应用
if __name__ == "__main__":
    gradio_interface().launch()
