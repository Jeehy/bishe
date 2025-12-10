# targets/planner_system.py
import json
from datetime import datetime
from deepseek_api import model_call
from executor import ToolExecutor
from prompt import (
    TASK_UNDERSTAND_PROMPT,
    PATH_PLANNER_PROMPT,
    PATH_EXECUTOR_PROMPT,
    REFLECTOR_PROMPT,
    STEP_DECIDER_PROMPT # <--- 确保引入了这个
)

# 自定义 JSON encoder 处理 MongoDB DateTime 等不可序列化的对象
class MongoDBJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # 处理 datetime 对象
        if isinstance(obj, datetime):
            return obj.isoformat()
        # 处理其他无法序列化的对象
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def escape_braces(text: str) -> str:
    """ 防止 Python .format() 把 JSON 的 {} 当成占位符 """
    return text.replace("{", "{{").replace("}", "}}")

def safe_parse_json(text):
    if not text:
        return {}
    
    # 清洗 Markdown 代码块标记
    cleaned_text = text.strip()
    if "```" in cleaned_text:
        import re
        # 匹配 ```json ... ``` 或 ``` ... ``` 中间的内容
        match = re.search(r"```(?:\w+)?\s*(.*?)s*```", cleaned_text, re.DOTALL)
        if match:
            cleaned_text = match.group(1)
            
    try:
        return json.loads(cleaned_text)
    except Exception:
        # 适用于返回了 {"paths": [...]} 的情况
        try:
            s = text[text.find("{"): text.rfind("}")+1]
            return json.loads(s)
        except Exception:
            return {}


class PlannerSystem:
    def __init__(self):
        self.executor = ToolExecutor()
        self.available_tools = list(self.executor.tools.keys())

    def _llm(self, prompt: str) -> dict:
        raw = model_call(prompt)
        # 记录原始 LLM 输出以便调试（保留到文件）
        try:
            with open("llm_raw_outputs.txt", "a", encoding="utf-8") as fh:
                fh.write("=== PROMPT START ===\n")
                fh.write(prompt + "\n")
                fh.write("=== RAW OUTPUT START ===\n")
                fh.write(raw + "\n")
                fh.write("=== RAW OUTPUT END ===\n\n")
        except Exception:
            pass
        return safe_parse_json(raw)

    def understand_task(self, user_input: str):
        prompt = TASK_UNDERSTAND_PROMPT.format(available_tools=",".join(self.available_tools), user_input=user_input)
        return self._llm(prompt)

    def plan_paths(self, task_json: dict):
        prompt = PATH_PLANNER_PROMPT.format(available_tools=",".join(self.available_tools), task_json=json.dumps(task_json, ensure_ascii=False))
        return self._llm(prompt)

    def synthesize_path(self, path_spec: dict, intermediate: list, task_understanding: dict):
        """
        执行路径推理综合
        """
        cleaned_intermediate = []
        for item in intermediate:
            clean_item = item.copy()
            result = clean_item.get("result", {})
            
            # Omics 结果，只保留 Top 50，防止几千个基因塞爆 Prompt
            if isinstance(result, dict) and result.get("type") == "run_omics":
                omics_res = result.get("results", {})
                if isinstance(omics_res, dict):
                    clean_res = {
                        "type": "run_omics",
                        "summary": "Data truncated for LLM context window",
                        "n_significant": result.get("n_significant"),
                        "top_upregulated": omics_res.get("top_upregulated", [])[:50],   # 只取前50
                        "top_downregulated": omics_res.get("top_downregulated", [])[:50] # 只取前50
                    }
                    clean_item["result"] = clean_res
            
            # OpenTargets也防止列表过长
            elif isinstance(result, dict) and result.get("type") == "query_opentargets":
                 raw_list = result.get("results", [])
                 if isinstance(raw_list, list) and len(raw_list) > 50:
                     clean_item["result"] = {
                         "type": "query_opentargets",
                         "n_results": len(raw_list),
                         "top_results": raw_list[:50] # 只给 LLM 看前 50 个
                     }
            
            cleaned_intermediate.append(clean_item)

        # 构造 payload
        payload_dict = {
            "path_spec": path_spec,
            "intermediate_outputs": cleaned_intermediate, 
            "task_understanding": task_understanding
        }

        payload_str = json.dumps(payload_dict, ensure_ascii=False, indent=2, cls=MongoDBJSONEncoder)
        prompt = PATH_EXECUTOR_PROMPT.replace("{payload}", payload_str)
        response_dict = self._llm(prompt)
        if isinstance(response_dict, dict):
            return response_dict
        else:
            return {"candidate_targets": [], "error": "LLM response format invalid"}
    
    # === 新增: 动态决策步骤 ===
    def step_decide(self, history: list, available_tools: list) -> dict:
        """
        根据执行历史动态决定下一步动作 (支持参数注入)
        """
        # 简化 history 以节省 Token，但保留关键结果用于提取基因
        simple_history = []
        for h in history:
            res = h.get("result", {})
            step_info = {
                "step": h.get("step"),
                "status": "success" if not res.get("error") else "error",
            }
            # 如果是 Omics/OpenTargets，保留前几个结果供 LLM 参考提取
            if res.get("type") == "run_omics" and "results" in res:
                # 提取 Top 基因名供 LLM 看
                try:
                    top_up = [g["gene_id"] for g in res["results"].get("top_upregulated", [])[:10]]
                    step_info["result_preview"] = f"Top Up Genes: {top_up}"
                except:
                    pass
            simple_history.append(step_info)

        history_str = json.dumps(simple_history, ensure_ascii=False, cls=MongoDBJSONEncoder)
        tools_str = ",".join(available_tools)
        
        # 使用 prompt.py 中定义的新 Prompt
        prompt = STEP_DECIDER_PROMPT.format(
            context=history_str, 
            available_tools=tools_str
        )
        
        return self._llm(prompt)

    def _ensure_novelty_notes(self, synthesis: dict, history: list):
        """
        如果 LLM 返回的 synthesis 中缺少 novelty_notes，自动根据历史数据补充。
        
        判断规则（真正的 novel）：
        - novel=true：不在 OpenTargets top 30 中，但在 Omics 差异表达中排名靠前（top 20）
                    或 PubMed 文献 2-3 篇但 OpenTargets 知名度低
        - novel=false：在 OpenTargets top 20 中，且 PubMed 文献 >= 3 篇（已知靶点）
        """
        if not synthesis:
            return synthesis
        if synthesis.get("novelty_notes"):
            return synthesis
        candidates = synthesis.get("candidate_targets", [])
        if not candidates:
            return synthesis
        
        # 查找历史中的 OpenTargets 结果（获取排名和分数）
        opentargets_rank = {} 
        for entry in history:
            result = entry.get("result", {})
            if result.get("type") == "query_opentargets":
                results = result.get("results", [])
                for idx, r in enumerate(results):
                    sym = r.get("symbol") or r.get("name")
                    score = r.get("score", 0)
                    if sym:
                        opentargets_rank[sym.upper()] = (idx, score)
                break
        
        # 查找历史中的 Omics 结果（获取排名）
        omics_rank = {} 
        for entry in history:
            result = entry.get("result", {})
            if result.get("type") == "run_omics":
                omics_results = result.get("results", {})
                if isinstance(omics_results, dict):
                    top_up = omics_results.get("top_upregulated", [])
                    top_down = omics_results.get("top_downregulated", [])
                    for idx, g in enumerate(top_up[:30]):  # 记录 top 30
                        gid = g.get("gene_id") or g.get("gene")
                        if gid:
                            omics_rank[gid.upper()] = (idx, "upregulated")
                    for idx, g in enumerate(top_down[:30]):
                        gid = g.get("gene_id") or g.get("gene")
                        if gid:
                            omics_rank[gid.upper()] = (idx + 30, "downregulated")
                break
        
        # 查找历史中的 PubMed 结果
        pubmed_count = 0
        for entry in history:
            result = entry.get("result", {})
            if result.get("type") in ("search_pubmed_mongo", "search_pubmed"):
                pubmed_count = result.get("n_results", 0)
                break
        
        # 为每个候选基因判断 novel 状态
        novelty_notes = {}
        for gene in candidates:
            if isinstance(gene, dict):
                gene_name = gene.get("gene") or gene.get("symbol") or str(gene)
            else:
                gene_name = str(gene)
            
            gene_upper = gene_name.upper()
            
            # 获取该基因在 OpenTargets 中的排名
            ot_info = opentargets_rank.get(gene_upper)
            ot_rank = ot_info[0] if ot_info else float('inf')
            # 获取该基因在 Omics 中的排名
            omics_info = omics_rank.get(gene_upper)
            omics_rank_idx = omics_info[0] if omics_info else float('inf')
            omics_direction = omics_info[1] if omics_info else None
            
            is_novel = True 
            reason_parts = []
            
            if ot_rank <= 20 and pubmed_count >= 3:
                # 明显的已知靶点
                is_novel = False
                reason_parts.append(f"OpenTargets 排名 {ot_rank+1}（top 20）")
                reason_parts.append(f"PubMed {pubmed_count} 篇")
            elif ot_rank > 30 and omics_rank_idx <= 20:
                # 明显的新颖候选
                is_novel = True
                reason_parts.append(f"OpenTargets 排名 {ot_rank+1}（> 30，低知名）")
                reason_parts.append(f"Omics 排名 {omics_rank_idx+1}（top 20，差异显著）")
            elif 30 < ot_rank <= 50 and pubmed_count <= 3:
                # 中等 OpenTargets 排名 + 文献不足 → 新颖
                is_novel = True
                reason_parts.append(f"OpenTargets 排名 {ot_rank+1}（30-50，中档知名）")
                reason_parts.append(f"PubMed {pubmed_count} 篇（证据有限）")
            elif ot_rank <= 10:
                # Top 10 的基本上都是已知
                is_novel = False
                reason_parts.append(f"OpenTargets 排名 {ot_rank+1}（top 10，高知名）")
            elif omics_rank_idx <= 10:
                # Omics top 10 但 OpenTargets 低排名 → 新颖
                is_novel = True
                reason_parts.append(f"Omics 排名 {omics_rank_idx+1}（top 10，数据驱动）")
                reason_parts.append(f"OpenTargets 排名 {ot_rank+1}（认可度低）")
            
            # 补充信息
            if omics_direction:
                reason_parts.append(f"方向: {omics_direction}")
            if pubmed_count > 0:
                reason_parts.append(f"文献: {pubmed_count} 篇")
            
            reason = " | ".join(reason_parts)
            if is_novel:
                reason += " → 新颖"
            else:
                reason += " → 已知"
            
            novelty_notes[gene_name] = {
                "novel": is_novel,
                "reason": reason
            }
        
        synthesis["novelty_notes"] = novelty_notes
        return synthesis

    def reflect_paths(self, paths_results: list, context_playbook: str = "none"):
        prompt = REFLECTOR_PROMPT.format(paths_results=json.dumps(paths_results, ensure_ascii=False, cls=MongoDBJSONEncoder), context_playbook=context_playbook)
        return self._llm(prompt)

    def execute_path_with_reflection(self, path_spec: dict, task_json: dict, logs: list):
        steps = list(path_spec.get("steps", []))
        history = [] 
        i = 0
        max_iter = max(200, len(steps)*10)
        iter_count = 0

        while i < len(steps) and iter_count < max_iter:
            iter_count += 1
            step_item = steps[i]
            
            # 统一处理: step_item 可能是字符串，也可能是我们插入的字典(包含 args)
            current_step_name = ""
            current_step_args = {}
            
            if isinstance(step_item, dict):
                current_step_name = step_item.get("tool")
                current_step_args = step_item.get("args", {})
            else:
                current_step_name = step_item
                current_step_args = {}

            # 遇到 <decide> 占位符，或者每一步执行前都进行一次 check (取决于你的策略，这里假设只有 <decide> 触发决策)
            # 为了更智能，我们可以在每一步之后都 evaluate，或者只在占位符处 evaluate。
            # 这里保持原逻辑：如果是占位符，或者需要在运行时动态插入
            
            # --- 动态决策逻辑 Start ---
            # 如果是 <decide>，或者是正常的步骤但你想让它有机会"插入"验证
            need_decision = False
            if isinstance(current_step_name, str) and current_step_name.startswith("<"):
                need_decision = True
            
            # 这里的逻辑是：如果是占位符，强制决策；否则执行完一步后，也可以决策（在下面）
            if need_decision:
                decision_json = self.step_decide(history, self.available_tools)
                dec_type = decision_json.get("decision", "CONTINUE")
                
                logs.append({"type":"decide", "decision": decision_json}) # 记录完整决策

                if dec_type == "STOP":
                    break
                
                elif dec_type == "INSERT":
                    tool_name = decision_json.get("tool")
                    tool_args = decision_json.get("args", {})
                    reason = decision_json.get("reason", "")
                    
                    if tool_name:
                        # 构造一个带参数的步骤对象插入队列
                        new_step_obj = {"tool": tool_name, "args": tool_args, "reason": reason}
                        steps.insert(i+1, new_step_obj) # 插入到当前位置之后
                        # 移除当前的 <decide> (如果是占位符的话)
                        if current_step_name.startswith("<"):
                            steps.pop(i) 
                            # i 不变，下次循环执行新插入的步骤
                            continue
                        else:
                            # 这种情况应该不会发生，因为我们只在 need_decision=True 时进来
                            pass

                elif dec_type == "CONTINUE":
                    if current_step_name.startswith("<"):
                        steps.pop(i) # 移除占位符，继续下一个
                        continue
                
                # 处理 SWITCH 等其他逻辑...
                # ...
            # --- 动态决策逻辑 End ---

            # 如果当前是占位符且选择了 CONTINUE，上面已经 continue 了。
            # 下面是执行实际步骤
            
            if not current_step_name or current_step_name.startswith("<"):
                 i+=1
                 continue

            # === 关键修改：参数传递 ===
            # 构造执行上下文
            # 注意：我们需要把 args 传给 executor。
            # 由于 executor.execute 的签名通常固定，我们通过 task_context 传递额外参数
            # 或者修改 executor.py (推荐)。
            # 这里采用一种兼容性写法：把 args 放到 task_context 的顶层，工具层通常会从 context.get("genes") 取
            
            task_context = {"task": task_json}
            # 将 args 注入 context (例如 genes, query 等)
            if current_step_args:
                task_context.update(current_step_args)

            logs.append({"type":"executing", "step": current_step_name, "args": current_step_args})
            
            # 执行
            tool_output = self.executor.execute(current_step_name, task_context, history=history)
            
            # ... (后续日志处理保持不变) ...
            summary = {"step": current_step_name, "type": tool_output.get("type") if isinstance(tool_output, dict) else str(type(tool_output)), "brief": None}
            if isinstance(tool_output, dict):
                 # ... (保持原有的 summary 生成逻辑) ...
                 pass

            history.append({"step": current_step_name, "args": current_step_args, "result": tool_output})
            logs.append({"type":"step", "step": current_step_name, "summary": summary})

            # 每一步结束后，再次给机会进行决策 (Post-Step Decision)
            # 这让 Agent 能够看到结果后立即反应（例如 Omics 跑完，马上决定插入 KG 验证）
            decision_json = self.step_decide(history, self.available_tools)
            dec_type = decision_json.get("decision", "")
            
            if dec_type == "INSERT":
                tool_name = decision_json.get("tool")
                tool_args = decision_json.get("args", {})
                new_step_obj = {"tool": tool_name, "args": tool_args, "reason": "post_step_insert"}
                steps.insert(i+1, new_step_obj)
                logs.append({"type":"auto_insert", "inserted": tool_name, "args": tool_args})
            elif dec_type == "STOP":
                break
            
            i += 1

        # ... (Synthesis 逻辑保持不变) ...
        synthesis = self.synthesize_path(path_spec, history, task_json)
        synthesis = self._ensure_novelty_notes(synthesis, history) 
        logs.append({"type":"synthesis", "path_id": path_spec.get("path_id"), "synthesis": synthesis})

        return {"path_id": path_spec.get("path_id"), "history": history, "synthesis": synthesis, "steps": steps}