import gradio as gr
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


# 选择嵌入模型的函数
def select_embedding_model(model_name: str):
    if model_name == "all-MiniLM-L6-v2":
        print("使用 all-MiniLM-L6-v2 模型...")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    elif model_name == "text2vec-base-chinese":
        print("使用 text2vec-base-chinese 模型...")
        return HuggingFaceEmbeddings(model_name="text2vec-base-chinese")
    else:
        raise ValueError("未识别的模型名称，请选择有效的模型（'all-MiniLM-L6-v2' 或 'text2vec-base-chinese'）")


# 增量更新 FAISS 向量库
def ingest_docs(model_name="all-MiniLM-L6-v2", data_dir="data", save_dir="vectorstore"):
    # 选择嵌入模型
    embedding = select_embedding_model(model_name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    # 加载现有的 FAISS 向量数据库
    if os.path.exists(save_dir):
        print(f"加载现有的 FAISS 向量数据库...")
        db = FAISS.load_local(save_dir, embedding, allow_dangerous_deserialization=True)  # 允许危险的反序列化
    else:
        print(f"未找到现有向量数据库，正在创建新的数据库...")
        db = FAISS(embedding)
    all_docs = []

    for filename in os.listdir(data_dir):
        if not filename.endswith((".pdf", ".txt", ".md", ".docx")):
            continue
        file_path = os.path.join(data_dir, filename)
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        splits = splitter.split_documents(docs)

        # 添加 metadata：记录文件名
        for doc in splits:
            doc.metadata["source"] = filename
        all_docs.extend(splits)

    # 将新文档添加到现有数据库
    db.add_documents(all_docs)

    # 保存更新后的数据库
    db.save_local(save_dir)
    print(f"✅ 向量库已更新，共索引 {len(all_docs)} 段内容。")


# 创建 Gradio 界面
def gradio_interface():
    # 模型选择下拉框
    model_options = ["all-MiniLM-L6-v2", "text2vec-base-chinese"]

    # 使用 Gradio 的 Dropdown 输入框选择模型
    model_dropdown = gr.Dropdown(
        choices=model_options,
        label="选择嵌入模型",
        value="all-MiniLM-L6-v2"  # 默认模型
    )

    # 按钮：开始处理文档
    process_button = gr.Button("生成向量数据库")

    # 输出结果
    output = gr.Textbox(label="生成状态")

    # 使用 Blocks 构建界面并绑定按钮事件
    with gr.Blocks() as demo:
        # 在 Blocks 上下文中使用事件绑定
        model_dropdown.render()
        process_button.render()
        output.render()

        process_button.click(
            fn=ingest_docs,
            inputs=[model_dropdown],
            outputs=[output]
        )

    return demo


# 启动 Gradio 应用
if __name__ == "__main__":
    gradio_interface().launch()
