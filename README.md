# Dog_Healthy_Diet_RAG
通过 RAG（检索增强生成） 技术，本项目为 宠物狗健康饮食咨询 提供了一个自动化问答系统。通过 FAISS 向量数据库和 Ollama 模型，系统能够结合相关文档生成回答，并通过 Gradio Web 界面支持用户交互，同时保持连续对话的上下文连贯性。
# Dog Healthy Diet RAG - 自制宠物狗健康饮食问答系统

## 项目概述
本项目旨在为宠物狗提供健康的自制饮食，利用 **检索增强生成（RAG）** 技术和大语言模型（LLM），通过输入问题，系统可以根据提供的文档生成准确的答案。

### 功能特点：
- **基于 RAG 的问答系统**：结合文档检索和生成模型，为用户提供个性化的饮食食谱建议。
- **多种模型支持**：支持选择不同的词嵌入模型（如 all-MiniLM-L6-v2 和 text2vec-base-chinese）和基底模型（如 DeepSeek-R1 和 LLaMA）。
- **文档检索**：通过 FAISS 向量数据库对上传的文档进行检索，以提供高效的答案生成。
- **连续对话功能**：支持上下文记忆，模拟真实对话的能力。
- **可调整的检索参数**：支持设置 `k`（返回前 k 个文档）、`nprobe` 和 `efSearch` 来优化检索效果。

## 技术栈
- **LangChain**：用于实现文档检索和生成问答链。
- **Ollama**：作为基底模型生成回答。
- **FAISS**：用于高效存储和检索文档向量。
- **HuggingFaceEmbeddings**：用于生成文档和问题的嵌入向量。
- **Gradio**：提供简洁的 Web 界面，支持用户交互。

## 目录结构

```plaintext
/data                  # 存放输入的文本文件、PDF、Markdown等文档
/models                # 存放预训练模型
/vectorstore           # 存储向量数据库
/flagged               # 标记需要处理的文档
/ingest.py             # 用于文档加载和向量数据库构建
/qa.py                 # 处理用户查询并生成答案
/requirements.txt      # 项目依赖
/README.md             # 项目的说明文件```


### 安装与使用
## 安装依赖
#克隆本项目到本地：
git clone https://github.com/daliJIAO/Dog_Healthy_Diet_RAG.git
cd Dog_Healthy_Diet_RAG
#安装项目所需的 Python 库：
pip install -r requirements.txt
#配置模型
下载并配置 HuggingFace 模型（如 all-MiniLM-L6-v2 或 text2vec-base-chinese）。

配置 FAISS 向量数据库，并确保 ingest.py 中的嵌入模型与实际使用的模型保持一致。

###  启动项目
使用 ingest.py 加载文档并构建 FAISS 向量数据库：
python ingest.py
## 启动问答系统界面：
python qa.py
该命令将启动一个 Gradio Web 界面，你可以通过浏览器访问并与系统交互。

## 调整检索参数
在 Gradio 界面中，用户可以调整以下参数：

#历史轮数：限制对话历史的最大长度，避免模型处理过长的历史数据。

#k：返回前 k 个最相关的文档，值越大，结果越全面，但响应时间可能增加。

#nprobe：搜索时使用的簇数，增加它可以提高召回率。

#efSearch：HNSW 搜索时的查询广度，增大它可提高检索的精度。

### 项目贡献
如果你对本项目有任何建议或贡献，欢迎提交 issue 或 pull request。

### License
本项目使用 MIT 许可协议，详情请查看 LICENSE 文件。
---

### 说明：
1. **项目概述**：概括性介绍了该项目的功能及技术栈。
2. **目录结构**：列出了项目的文件结构，帮助用户理解项目组织。
3. **安装与使用**：详细描述了如何设置和运行项目，包括安装依赖、配置模型以及启动服务的步骤。
4. **调整检索参数**：介绍了 Gradio 界面中可调整的关键参数，并解释了每个参数的作用。
5. **贡献和许可证**：提供了如何贡献代码的信息，并说明了项目的许可证类型。

通过该 `README.md` 文件，其他开发者和用户可以快速理解项目的目的、安装过程、使用方式以及如何参与贡献。如果你有更多信息或者需要修改的地方，可以随时告诉我！
