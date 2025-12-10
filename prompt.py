# prompt.py
# 所有 Prompt 模板（few-shot 示例 + 严格 JSON 要求）
# 已修复所有 { } 冲突问题，并增强了 Path Executor 的策略引导

TASK_UNDERSTAND_PROMPT = """
任务理解（Task Understanding）
系统可用工具/数据库: [{available_tools}]
目标：本次任务优先发现 **未经现有数据库/文献验证（novel/unvalidated）** 的潜在靶点，并给出起始动作建议。
请把用户自然语言输入转换为结构化 JSON，严格输出 JSON，字段说明：
- topic: 主题（如 "liver cancer"）
- goal: 目标（如 "novel_target_discovery"）
- seek_novel_targets: 布尔，是否优先寻找未经验证的靶点
- known_databases: 你判断可用的数据库/知识源（必须从上面系统可用工具中选择）
- suggested_start: 优先起始动作（从上面工具中选择）
- reason: 简短理由

示例：
输入: "请尝试发现潜在的肝癌靶点，优先那些未被数据库标注或没有强文献支持的"
输出:
{{
   "topic": "liver cancer",
   "goal": "novel_target_discovery",
   "seek_novel_targets": true,
   "known_databases": ["OpenTargets","MongoDB"],
   "suggested_start": ["query_opentargets","run_omics"],
   "reason": "OpenTargets 提供已知靶点，组学用于验证；目标是筛出未在 OpenTargets 或本地文献中被充分验证的候选"
}}
现在请将下面用户输入解析为 JSON（仅输出 JSON）：
{user_input}
"""

PATH_PLANNER_PROMPT = """
路径生成（Path Planner）
输入 task_understanding JSON，并且系统可用工具: [{available_tools}]
请生成 2~4 条候选路径（JSON 数组）。每条路径对象包含：
- path_id: 唯一 id
- steps: 动作列表（可选动作示例：query_opentargets, run_omics, query_kg, query_mongo_local, query_gene, web_search）
   允许使用特殊占位符 "<decide>" 表示在运行时由系统/LLM动态决定后续动作（例如 插入 query_gene 或 query_kg）。
- reason: 简短理由

示例输出:
[
   {{
      "path_id":"p1",
      "steps":["query_opentargets","run_omics","<decide>","query_mongo_local"],
      "reason":"先利用已知靶点再验证表达，动态检查 novel 候选，最后通过本地文献验证其研究稀缺度"
   }},
   {{
      "path_id":"p2",
      "steps":["run_omics","query_kg","<decide>","query_mongo_local"],
      "reason":"数据驱动后检索图谱，最后使用本地 Mongo 文献挖掘确认其辅助证据"
   }}
]

输入 task_understanding:
{task_json}
"""

PATH_EXECUTOR_PROMPT = """
推理综合（Path Executor）

目标：请优先标注并推荐那些 **未经数据库/文献验证（novel / unvalidated）** 的潜在靶点。

*** IMPORTANT SYNTHESIS STRATEGY (至关重要) ***
1. **并集策略 (Union Strategy)**: 不要局限于寻找在所有工具中都出现的基因。不要只找交集。
2. **数据驱动 (Data-Driven)**: 如果 Omics 数据显示某基因显著差异（Top 50），即使 KG 或 OpenTargets 没有记录，**必须**将其选入候选！这正是我们要找的 "Novel Target"。
3. **避免保守**: 不要只输出 'TP53' 等众所周知的靶点。请尽可能挖掘 **5-8 个** 候选基因。
4. **新颖性**: 优先选择不在 OpenTargets Top 20 列表中，但在 Omics 或 KG 中有强证据的基因。

下面是你需要参考的输入内容（Payload）：
{payload}

你必须输出严格 JSON（无额外文本），字段要求：
- reasoning_chain: 字符串，简述你如何结合数据筛选出下面候选的
- candidate_targets: 字符串数组，至少列出 5 个基因名（混合已知和新颖）
- novelty_notes: 字典对象，格式为 {{ "GeneName": {{ "novel": bool, "reason": "例如：Omics Top 3 但 OpenTargets 未收录" }} }}
- confidence: 0-1 之间浮点数
- new_queries: 数组或空数组
- change_path: true/false
"""


REFLECTOR_PROMPT = """
反思器（Reflector）
输入：多条路径的结果（paths_results）与上下文（context_playbook）。
请判断：
- 是否存在共识靶点（consensus），并说明这些共识是否 novel（未经验证）
- 是否已经收敛（converged）
- 若未收敛，提出 1~3 条新的建议路径（suggested_paths），优先探索 novel 证据薄弱的候选
- 以及建议的新检索（new_queries）

严格输出 JSON，例如：
{{
   "consensus": ["TP53"],
   "consensus_novel": {{"TP53": false}},
   "converged": false,
   "suggested_paths": [
      {{
         "path_id":"p3",
         "steps":["query_mongo_local","run_omics"],
         "reason":"首先通过本地文献确认研究空白，再结合组学验证"
      }}
   ],
   "new_queries": ["验证 TP53 在 MongoDB 文献中的研究强度"]
}}

paths_results:
{paths_results}

context_playbook:
{context_playbook}
"""

# bishe/prompt.py

# ... (其他 Prompt 保持不变) ...

STEP_DECIDER_PROMPT = """
Step Decider（运行时决策）
你是一个智能执行决策器。你的目标是根据执行历史，决定下一步的最佳行动。

【当前上下文】
{context}

【可用工具】
[{available_tools}]

【决策逻辑】
1. **检查失败**: 如果上一步失败或为空，尝试换个工具或修改查询。
2. **数据驱动验证 (关键)**: 
   - 如果历史记录(如 run_omics)发现了显著基因(Top list)，且尚未在 KG/OpenTargets 中验证，请**立即插入**验证步骤。
   - 此时必须从历史中提取基因名，填入 `args` 字段。
3. **继续或停止**: 如果一切顺利且无需额外操作，选择 CONTINUE；如果信息已充足，选择 STOP。

【输出格式】
请严格输出 JSON 对象（不要Markdown格式）：

Case 1: 插入新步骤（带参数）
{{
    "decision": "INSERT",
    "tool": "query_kg",
    "args": {{ "genes": ["PAGE2", "MAGEA4", "TP53"] }},
    "reason": "Omics 发现了这些高表达基因，需要专门验证其图谱关联"
}}

Case 2: 插入新步骤（无参数）
{{
    "decision": "INSERT",
    "tool": "web_search",
    "args": {{ "query": "liver cancer novel targets reviews 2024" }},
    "reason": "需要补充最新的综述信息"
}}

Case 3: 继续原计划
{{
    "decision": "CONTINUE",
    "reason": "上一步成功，继续执行原定流程"
}}

Case 4: 停止
{{
    "decision": "STOP",
    "reason": "已找到足够候选，无需继续"
}}
"""