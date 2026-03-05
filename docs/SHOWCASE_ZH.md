# Eidetic 使用说明（中文）

<p align="center">
  <img src="../assets/logo/eidetic-logo.svg" alt="Eidetic Logo" width="860" />
</p>

<p align="center"><b>统一接口接入多种 Agent Memory，按需加载，低耦合切换，内置持久化。</b></p>

---

## 这个包解决什么问题

不同 memory system 的接口和数据结构差异很大，导致：

- 业务代码和底层 memory 强绑定
- 切换 memory 后端时改造成本高
- 多后端并存时测试与维护复杂

Eidetic 的目标是提供**统一、稳定、可替换的 memory 抽象层**，让你在不改业务逻辑的前提下自由切换 memory 系统。

---

## 核心能力

- **统一 API：** `ingest / remember / recall / forget / compact`
- **4 种运行模式：** `mock → persistent → native → auto`（智能降级）
- **内置 SQLite 持久化后端：** 零额外依赖，支持 FTS5 全文检索
- **4 个原生后端：** Letta、Microsoft GraphRAG、LightRAG（港大）、HippoRAG — 均已完整实现
- **插件懒加载：** 只在 `create(system=...)` 时加载后端，按需引入
- **Async 优先 + Sync 包装：** 兼容服务端和脚本场景
- **LangChain 官方适配层**

---

## 持久化模式

每个插件均支持四种模式，通过 `config["plugin_config"]["mode"]` 选择：

| 模式 | 存储位置 | 需要外部依赖 | 适用场景 |
|---|---|---|---|
| `mock` | 纯内存（进程退出即消失） | 无 | 单元测试、快速原型 |
| `persistent` | SQLite 磁盘文件 | 无（stdlib） | 开发环境、单进程生产 |
| `native` | 后端自带存储 | 需要对应后端库 | 完整生产环境 |
| `auto`（**默认**） | 有后端库用 native，否则 SQLite | 可选 | **推荐默认值** |

在 `auto` 模式下，若后端库未安装，Eidetic 会自动降级到内置 SQLite 后端并发出 `UserWarning`，代码无需修改即可运行。

---

## 安装

```bash
# 基础包（含内置 SQLite 后端）
pip install eidetic

# 各后端插件（按需安装）
pip install "eidetic[letta]"      # 需要运行 Letta Server
pip install "eidetic[graphrag]"   # Microsoft GraphRAG
pip install "eidetic[lightrag]"   # LightRAG（港大），需要 LLM API
pip install "eidetic[langchain]"  # LangChain 适配层

# HippoRAG 暂无稳定 PyPI 包，从 GitHub 安装
pip install "git+https://github.com/OSU-NLP-Group/HippoRAG.git"
```

---

## 基本使用

### Mock 模式（测试用，无任何依赖）

```python
from eidetic import MemoryManager, Document, MemoryEvent, RecallQuery, ForgetRequest

manager = MemoryManager()
memory = manager.create("letta", config={"plugin_config": {"mode": "mock"}})

memory.ingest([Document(content="Eidetic 统一了 Agent 的 memory 接口", tags=["intro"])])
memory.remember(MemoryEvent(content="用户偏好简洁回答", role="user", tags=["profile"]))

result = memory.recall(RecallQuery(query="memory 接口", top_k=3))
print([item.content for item in result.items])

memory.forget(ForgetRequest(tags=["profile"], hard_delete=True))
memory.compact()
```

### Persistent 模式（SQLite 持久化，零依赖）

```python
manager = MemoryManager()
memory = manager.create(
    "letta",
    config={"plugin_config": {"mode": "persistent", "db_path": "agent_memory.db"}},
)

# 数据在进程重启后仍然保留
memory.ingest([Document(content="项目启动日期：2025-01-10", tags=["project"])])
result = memory.recall(RecallQuery(query="项目启动"))
```

### Native 模式（真实后端）

```python
# Letta — 需要先启动 Letta Server：letta server
memory = manager.create(
    "letta",
    config={
        "plugin_config": {
            "mode": "native",
            "base_url": "http://localhost:8283",
            "agent_name": "my-agent",
        }
    },
)

# LightRAG — 需要 OpenAI（或 Ollama）API Key
memory = manager.create(
    "lightrag",
    config={
        "plugin_config": {
            "mode": "native",
            "working_dir": "./lightrag_data",
            "llm_provider": "openai",       # 或 "ollama"、"anthropic"、"custom"
            "llm_model": "gpt-4o-mini",
        }
    },
)

# GraphRAG — 批量索引，写入文件后需手动触发建索引
memory = manager.create(
    "graphrag",
    config={
        "plugin_config": {
            "mode": "native",
            "working_dir": "./graphrag_data",
            "query_mode": "local",          # 或 "global"
        }
    },
)
# 写入文档后，通过逃生舱触发建索引
await memory.async_handle.backend.build_index()
```

### Async 接口

```python
import asyncio
from eidetic import MemoryManager, RecallQuery

async def main():
    manager = MemoryManager()
    memory = await manager.acreate("letta", config={"plugin_config": {"mode": "persistent"}})
    result = await memory.recall(RecallQuery(query="anything"))
    print(result.items)

asyncio.run(main())
```

---

## 后端对比

| 系统 | 安装 extras | 持久化 | 需要 LLM | 图谱检索 | 说明 |
|---|---|---|---|---|---|
| **SQLite**（内置） | — | ✅ WAL 磁盘 | ✗ | ✗ | 默认降级目标，FTS5 全文检索 |
| **Letta** | `eidetic[letta]` | ✅ 服务端 DB | ✗ | ✗ | 需要运行 `letta server` |
| **GraphRAG** | `eidetic[graphrag]` | ✅ 磁盘索引 | ✅ | ✅ | 批量管道，ingest 后需调用 `build_index()` |
| **LightRAG** | `eidetic[lightrag]` | ✅ working\_dir | ✅ | ✅ | `recall` 返回综合答案字符串（非原始片段） |
| **HippoRAG** | GitHub 安装 | ✅ save\_dir | ✅ | ✅ | 每次 ingest 都会重建索引，适合读多写少场景 |

---

## 统一接口与特异接口

**推荐默认：使用统一接口，保持可移植性**

```python
memory.ingest(...)
memory.remember(...)
memory.recall(...)
memory.forget(...)
memory.compact()
```

切换后端只需改 `system=` 参数，业务代码不变。

**高级用法：通过逃生舱访问后端原生能力**

```python
native_backend = memory.async_handle.backend

# GraphRAG：触发建图索引（耗时操作）
await native_backend.build_index()

# LightRAG：删除文档后重建图谱（purge soft-deleted）
await native_backend.rebuild_index()
```

使用逃生舱会增加对具体后端的耦合，请酌情使用。

---

## LangChain 集成

```python
from eidetic.integrations.langchain import EideticLangChainMemory

# Letta 未安装时自动降级到 SQLite
memory = EideticLangChainMemory(
    system="letta",
    config={"plugin_config": {"mode": "persistent", "db_path": "chat.db"}},
    input_key="input",
    memory_key="history",
    session_tag="session-42",
    top_k=5,
)
```

---

## 常见问题

| 报错 | 原因 | 解决方案 |
|---|---|---|
| `DependencyMissingError` | 后端库未安装 | 按报错中的 `install_hint` 安装 |
| `PluginNotFoundError` | `system=` 拼写错误 | 使用 `manager.list_systems()` 查看可用名称 |
| `RuntimeError: Cannot call ... inside event loop` | 在 async 环境中使用了 sync API | 改用 `await manager.acreate(...)` 和 `await memory.recall(...)` |
| GraphRAG `recall` 返回空 | 尚未建索引 | 先调用 `await backend.build_index()` |
| LightRAG `recall` 返回 None | 图谱尚未建立 | 确保已 `ingest` 足够文档后再查询 |

---

## 测试

```bash
python -m pytest -q
```

测试覆盖全部四个插件系统的 `mock` 模式，验证 `ingest → remember → recall → forget → compact` 完整生命周期、错误语义及 LangChain 适配层。

---

## 参考链接

- 英文主说明：[README.md](../README.md)
- Notebook 示例（统一接口 vs 特异接口）：
  [common\_vs\_specific\_interfaces.ipynb](../examples/notebooks/common_vs_specific_interfaces.ipynb)
- 试用脚本：[try\_eidetic.py](../examples/try_eidetic.py)
