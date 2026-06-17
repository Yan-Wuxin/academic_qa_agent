# Academic QA Agent

> 一个面向学术论文问答场景的 Agentic RAG 项目（Phase 1）。

该项目旨在构建一个能够理解学术论文、支持复杂问题检索与回答的 Academic AI Assistant。

目前已完成第一阶段（Advanced RAG），后续将逐步升级为基于 LangGraph 的 Multi-Agent Research Assistant。

---

# 项目特色

相比传统 RAG，本项目实现了完整的高级检索增强流程：

- Query Rewrite

- HyDE（Hypothetical Document Embedding，当前已实现，可配置启用）

- Hybrid Retrieval（Vector + BM25）

- Cross Encoder Reranker

- Tool Routing

- 多轮对话 Memory

- 本地 Embedding / 本地 Reranker

项目整体采用模块化设计，为后续 Agent Workflow 与 LangGraph 编排提供基础。

---

# 系统架构

```text
                User Query
                     │
                     ▼
              Tool Router
                     │
        ┌────────────┴────────────┐
        │                         │
     普通聊天                  Academic QA
                                  │
                                  ▼
                         Query Rewrite
                                  │
                           (Optional HyDE)
                                  │
                                  ▼
                     Hybrid Retrieval
                  (Vector + BM25)
                                  │
                                  ▼
                         CrossEncoder
                           Reranker
                                  │
                                  ▼
                          Prompt Builder
                                  │
                                  ▼
                              LLM Answer
```

---

# 当前功能

## 1. 文档处理

支持：

- PDF
- Markdown（可扩展）

包括：

- 文档加载
- 文本切分
- Embedding
- ChromaDB 向量存储

---

## 2. Query Transform

当前支持：

- Query Rewrite
- HyDE（可选）

用于优化检索效果。

---

## 3. Hybrid Retrieval

采用：

- Dense Retrieval（Embedding）
- Sparse Retrieval（BM25）

融合检索结果，提高召回率。

---

## 4. Reranker

采用：

BAAI/bge-reranker-base

进行二阶段排序。

相比纯向量检索，可显著提升最终命中质量。

---

## 5. Tool Routing

根据用户问题自动选择：

- 普通聊天
- RAG问答

减少不必要的检索，提高响应速度。

---

## 6. Memory

支持多轮对话上下文管理。

---

# 项目结构

```text
academic_qa_agent
│
├── agent/                  # Agent执行入口
│
├── config/                 # 配置
│
├── core/
│
├── data/                   # 文档数据
│
├── evaluation/             # RAG评测（持续完善）
│
├── infra/
│   ├── loader_service.py
│   ├── splitter_service.py
│   ├── embedding_service.py
│   └── vector_store_service.py
│
├── memory/
│
├── model/
│   ├── bge-small-zh-v1.5
│   └── bge-reranker-base
│
├── prompts/
│
├── rag/
│   ├── query_transform.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── rag_pipeline.py
│   └── rag_service.py
│
├── tools/
│
└── utils/
```

---

# 检索流程

```text
Query
   │
Rewrite
   │
HyDE（Optional）
   │
Hybrid Retrieval
   │
Reranker
   │
Top-k Context
   │
Prompt
   │
LLM
```

---

# 技术栈

| 模块 | 技术 |
|------|------|
| LLM | DeepSeek / OpenAI Compatible API |
| Framework | LangChain |
| Embedding | BGE Small |
| Reranker | BGE Reranker |
| Vector DB | ChromaDB |
| Sparse Retrieval | BM25 |
| Retriever | Hybrid Retrieval |
| Memory | LangChain Memory |
| Language | Python 3.11 |

---

# 示例效果

示例论文：

> Attention Is All You Need

示例问题：

> Base Transformer 使用多少个 Attention Heads？

回答：

> Base Transformer 使用 8 个 Attention Heads。

---

问题：

> Encoder 和 Decoder 分别由哪些组件组成？

系统能够结合论文内容给出结构化回答。

---

# Roadmap

## Phase 1（已完成）

- [x] Query Rewrite
- [x] HyDE
- [x] Hybrid Retrieval
- [x] CrossEncoder Reranker
- [x] Tool Routing
- [x] Memory
- [x] ChromaDB

---

## Phase 2（开发中）

- [ ] LangGraph Workflow
- [ ] Planner Agent
- [ ] Retriever Agent
- [ ] Reader Agent
- [ ] Critic Agent
- [ ] Citation Grounding

---

## Phase 3

- [ ] Multi-Agent Research Assistant
- [ ] Automatic Literature Review
- [ ] RAG Evaluation Pipeline
- [ ] Report Generation
- [ ] Web Search Integration

---

# 项目目标

本项目将逐步演进为一个面向科研场景的 Agentic RAG 系统，重点探索：

- Agent Workflow
- Multi-Agent Collaboration
- Advanced Retrieval
- Citation Grounding
- RAG Evaluation
- Research Assistant

最终目标是构建一个能够完成论文阅读、检索、总结、分析与综述生成的 Academic AI Agent。

---

# License

MIT License
