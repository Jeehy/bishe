# prompt.py
# 所有 Prompt 模板（few-shot 示例 + 严格 JSON 要求）

# === 工具规格说明 (嵌入到 Prompt 中) ===
# 这部分文本会被硬编码到下方的 Prompt 中，指导 LLM 如何使用工具参数
TOOL_SPECS_TEXT = """
【工具详细规格说明 (Tool Specs)】
1. search_literature (文献证据检索 - 核心工具):
   - 功能: 从本地向量库检索文献证据。
   - **核心用法 (推荐)**: {{ "gene": "TP53" }} -> 系统会自动从"临床预后/分子机制/药物治疗"三个维度生成综述。
   - 备用用法: {{ "query": "liver cancer survival" }} -> 仅进行简单的语义搜索。
   - 适用场景: 验证靶点新颖性、查找机制支撑、分析药物潜力。

2. run_omics (组学分析):
   - 功能: 获取差异表达基因(DEGs)。
   - 参数: 通常无需参数，或 {{ "top_k": 50 }}。
   - 适用场景: 任务开始时发现数据驱动的候选。

3. query_opentargets (公共数据库):
   - 功能: 查询 OpenTargets 数据库中的已知关联评分。
   - 参数: {{ "genes": ["G1", "G2"] }} 或空 (查全量)。
   - 适用场景: 判断靶点是否“已知”，排除高知名度靶点。

4. query_kg (知识图谱):
   - 功能: 查询基因在图谱中的关联实体。
   - 参数: {{ "genes": ["G1", "G2"] }}。
"""

# 注意：以下均使用普通字符串 (无 f 前缀)，避免提前转义花括号

TASK_UNDERSTAND_PROMPT = """
任务理解（Task Understanding）
系统可用工具列表: [{available_tools}]

{tool_specs}

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
   "known_databases": ["query_opentargets", "search_literature"],
   "suggested_start": ["query_opentargets", "run_omics"],
   "reason": "OpenTargets 提供已知靶点，组学用于验证；目标是筛出未在 OpenTargets 或本地文献中被充分验证的候选"
}}
现在请将下面用户输入解析为 JSON（仅输出 JSON）：
{user_input}
"""

PATH_PLANNER_PROMPT = """
路径生成（Path Planner）
输入 task_understanding JSON，并且系统可用工具: [{available_tools}]

{tool_specs}

请生成 2~4 条候选路径（JSON 数组）。每条路径对象包含：
- path_id: 唯一 id
- steps: 动作列表。
  - **重要**: 如果某个步骤需要特定参数（如查询特定基因），请使用对象格式：{{"tool": "工具名", "args": {{ "key": "value" }} }}。
  - 如果不需要参数，可以直接用字符串 "工具名"。
  - 允许使用特殊占位符 "<decide>"。
- reason: 简短理由

示例输出:
[
   {{
      "path_id": "p1",
      "steps": [
          {{"tool": "search_literature", "args": {{"gene": "TP53"}} }},
          "run_omics",
          "<decide>"
      ],
      "reason": "用户明确指定查询TP53，因此第一步必须带参数调用；随后结合组学验证"
   }},
   {{
      "path_id": "p2",
      "steps": ["run_omics", "query_kg"],
      "reason": "无特定目标基因，先进行广撒网式的数据驱动挖掘"
   }}
]

输入 task_understanding:
{task_json}
"""

PATH_EXECUTOR_PROMPT = """
推理综合（Path Executor）

目标：请优先标注并推荐那些 **未经数据库/文献验证（novel / unvalidated）** 的潜在靶点。

*** IMPORTANT SYNTHESIS STRATEGY (至关重要) ***
1. **关注 search_literature 的 Summary**: 如果结果中包含 `search_literature`，请仔细阅读其 `summary` 字段。如果 Summary 显示“未找到相关证据”或证据很少，这反而可能是一个好的 **Novel** 信号！
2. **数据驱动 (Data-Driven)**: 如果 Omics 数据显示某基因显著差异（Top 50），即使 KG 或 OpenTargets 没有记录，**必须**将其选入候选！
3. **避免保守**: 不要只输出 'TP53' 等众所周知的靶点。请尽可能挖掘 **5-8 个** 候选基因。
4. **新颖性判断**: 
   - Known: OpenTargets 排名高 + search_literature 证据丰富。
   - Novel: Omics 显著 + (OpenTargets 无记录 OR search_literature 仅有少量机制研究无药物研究)。

下面是你需要参考的输入内容（Payload）：
{payload}

你必须输出严格 JSON（无额外文本），字段要求：
- reasoning_chain: 字符串，简述你如何结合数据筛选出下面候选的
- candidate_targets: 字符串数组，至少列出 5 个基因名（混合已知和新颖）
- novelty_notes: 字典对象，格式为 {{ "GeneName": {{ "novel": bool, "reason": "例如：Omics Top 3 但文献综述显示未见药物报道" }} }}
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
   "consensus_novel": {{ "TP53": false }},
   "converged": false,
   "suggested_paths": [
      {{
         "path_id":"p3",
         "steps":["search_literature","run_omics"],
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

STEP_DECIDER_PROMPT = """
Step Decider（运行时决策）
你是一个智能执行决策器。你的目标是根据执行历史，决定下一步的最佳行动。

【当前上下文】
{context}

【可用工具列表】
[{available_tools}]

{tool_specs}

【决策逻辑】
1. **检查失败**: 如果上一步失败或为空，尝试换个工具或修改查询。
2. **数据驱动验证 (关键 - Literature Loop)**: 
   - 如果历史记录(如 run_omics)发现了显著基因(Top list)，且尚未详细查证：
   - 请**立即插入** `search_literature` 步骤。
   - **必须**从结果中提取基因名，填入 `args` 的 `gene` 字段！这是获取多维证据的关键。
3. **继续或停止**: 如果一切顺利且无需额外操作，选择 CONTINUE；如果信息已充足，选择 STOP。

【输出格式】
请严格输出 JSON 对象（不要Markdown格式）：

Case 1: 插入文献验证（最常用）
{{
   "decision": "INSERT",
   "tool": "search_literature",
   "args": {{ "gene": "MAGEA4" }},
   "reason": "Omics 发现了 MAGEA4 高表达，需要调用 search_literature 获取其预后和药物机制综述"
}}

Case 2: 插入图谱查询
{{
   "decision": "INSERT",
   "tool": "query_kg",
   "args": {{ "genes": ["PAGE2", "TP53"] }},
   "reason": "需要验证这些基因的图谱关联"
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

# === 关键步骤：注入工具说明 ===
# 使用 .replace 将 tool_specs 注入到模板中，同时保留其他 {placeholder} 给 .format() 使用
TASK_UNDERSTAND_PROMPT = TASK_UNDERSTAND_PROMPT.replace("{tool_specs}", TOOL_SPECS_TEXT)
PATH_PLANNER_PROMPT = PATH_PLANNER_PROMPT.replace("{tool_specs}", TOOL_SPECS_TEXT)
STEP_DECIDER_PROMPT = STEP_DECIDER_PROMPT.replace("{tool_specs}", TOOL_SPECS_TEXT)