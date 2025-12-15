# targets/planner_system.py
import json, os
from datetime import datetime
from deepseek_api import model_call
from executor import ToolExecutor
from prompt import (
    TASK_UNDERSTAND_PROMPT, PATH_PLANNER_PROMPT,
    PATH_EXECUTOR_PROMPT, REFLECTOR_PROMPT, STEP_DECIDER_PROMPT
)

# === 辅助工具 ===
class MongoDBJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime): return obj.isoformat()
        try: return super().default(obj)
        except TypeError: return str(obj)

def safe_parse_json(text):
    if not text: return {}
    cleaned_text = text.strip()
    if "```" in cleaned_text:
        import re
        match = re.search(r"```(?:\w+)?\s*(.*?)s*```", cleaned_text, re.DOTALL)
        if match: cleaned_text = match.group(1)
    try:
        return json.loads(cleaned_text)
    except Exception:
        try:
            s = text[text.find("{"): text.rfind("}")+1]
            return json.loads(s)
        except Exception: return {}

# 提取逻辑独立，保持纯净
def extract_genes_from_result(tool_name: str, result: dict) -> list:
    found_genes = []
    try:
        if tool_name == "run_omics" and "results" in result:
            # 兼容 Discovery 模式的结构
            if "top_upregulated" in result["results"]:
                top_up = result["results"].get("top_upregulated", [])[:10]
                top_down = result["results"].get("top_downregulated", [])[:10]
                found_genes = [g.get("gene_id", g.get("gene")) for g in top_up + top_down]
            # 兼容 Verification 模式的结构 (如果验证了新基因，也可以视为线索)
            elif isinstance(result["results"], list):
                # 提取验证为显著的基因
                found_genes = [item["gene"] for item in result["results"] if item.get("is_significant")]
        elif tool_name == "query_kg":
            res_data = result.get("results", []) if isinstance(result, dict) else result
            if isinstance(res_data, list):
                found_genes = [item.get("name") or item.get("symbol") for item in res_data if isinstance(item, dict)]
        elif tool_name == "query_opentargets" and "results" in result:
            found_genes = [g.get("symbol") for g in result["results"][:10]]
    except Exception: pass
    return list(set([g for g in found_genes if g]))


class PlannerSystem:
    def __init__(self):
        self.executor = ToolExecutor()
        self.available_tools = list(self.executor.tools.keys())

    def _llm(self, prompt: str) -> dict:
        return safe_parse_json(model_call(prompt))

    # === 核心 Prompt 调用 ===
    def understand_task(self, user_input: str):
        return self._llm(TASK_UNDERSTAND_PROMPT.format(available_tools=",".join(self.available_tools), user_input=user_input))

    def plan_paths(self, task_json: dict):
        return self._llm(PATH_PLANNER_PROMPT.format(available_tools=",".join(self.available_tools), task_json=json.dumps(task_json, ensure_ascii=False)))

    def step_decide(self, history: list, available_tools: list) -> dict:
        simple_history = []
        for h in history:
            res = h.get("result", {})
            step_info = {"step": h.get("step"), "status": "success" if not res.get("error") else "error"}
            if res.get("type") == "run_omics" and "results" in res:
                try:
                    top_up = [g["gene_id"] for g in res["results"].get("top_upregulated", [])[:10]]
                    step_info["result_preview"] = f"Top Up Genes: {top_up}"
                except: pass
            simple_history.append(step_info)

        return self._llm(STEP_DECIDER_PROMPT.format(
            context=json.dumps(simple_history, ensure_ascii=False, cls=MongoDBJSONEncoder), 
            available_tools=",".join(available_tools)
        ))

    def reflect_paths(self, paths_results: list, context_playbook: str = "none"):
        return self._llm(REFLECTOR_PROMPT.format(paths_results=json.dumps(paths_results, ensure_ascii=False, cls=MongoDBJSONEncoder), context_playbook=context_playbook))

    # === 瘦身后的主执行循环 ===
    def execute_path_with_reflection(self, path_spec: dict, task_json: dict, logs: list):
        path_id = path_spec.get("path_id", "unknown_path")
        steps = list(path_spec.get("steps", []))
        history = [] 
        active_genes_bus = [] # 上下文总线
        searched_genes_history = set()
        evidence_dir = f"evidence_data/{path_id}"
        os.makedirs(evidence_dir, exist_ok=True)
        print(f"\n🚀 [Path: {path_id}] 开始执行，共 {len(steps)} 步...")

        i = 0
        while i < len(steps) and i < 50: # 防止死循环
            # 1. 动态决策 (Pre-Step)
            if self._handle_dynamic_decision(history, steps, i, logs, path_id, is_pre_step=True):
                if logs[-1].get("decision", {}).get("decision") == "STOP": 
                    break
                if i >= len(steps): break
                # 检查占位符
                step_item_check = steps[i]
                tool_name_check = step_item_check.get("tool") if isinstance(step_item_check, dict) else step_item_check
                if tool_name_check.startswith("<"): 
                     continue

            if i >= len(steps): break
            
            step_item = steps[i]
            tool_name = step_item.get("tool") if isinstance(step_item, dict) else step_item
            tool_args = step_item.get("args", {}) if isinstance(step_item, dict) else {}

            if not tool_name or tool_name.startswith("<"): 
                i+=1; continue

            # 2. 上下文总线：参数自动注入
            self._inject_context_genes(tool_name, tool_args, active_genes_bus, path_id)

            if tool_name == "search_literature":
                # 获取当前参数中的基因列表
                target_genes = tool_args.get("genes", [])
                if isinstance(target_genes, str): target_genes = [target_genes]
                
                # 过滤掉已经查过的基因
                new_genes = [g for g in target_genes if g not in searched_genes_history]
                
                # 如果有被过滤的，打印日志
                if len(new_genes) < len(target_genes):
                    skipped = set(target_genes) - set(new_genes)
                    print(f"     🧹 [Deduplication] 跳过已检索基因: {list(skipped)}")
                
                # 如果过滤后没有基因了，跳过此步
                if not new_genes:
                    print(f"     ⏭️ [Skip] 所有目标基因均已检索过文献，跳过此步。")
                    logs.append({"type": "skip", "step": tool_name, "reason": "duplicate_genes"})
                    i += 1
                    continue
                
                # 更新参数和历史记录
                tool_args["genes"] = new_genes
                searched_genes_history.update(new_genes)

            # 3. 执行工具
            logs.append({"type":"executing", "step": tool_name, "args": tool_args})
            print(f"  👉 [Path: {path_id}] [Step {i+1}] 执行: {tool_name} | 参数: {list(tool_args.keys())}")
            
            task_context = {"task": task_json, **tool_args}
            tool_output = self.executor.execute(tool_name, task_context, history=history)

            # 4. 上下文总线：结果捕获
            # 4. 上下文总线：结果捕获
            # 🆕 [Fix] 验证模式锁定：防止 OpenTargets 等工具返回的关联基因干扰主线
            if str(path_id).startswith("verify_"):
                print(f"     🔒 [Path: {path_id}] [Bus] 验证模式")
            else:
                # 只有非验证模式（发现模式）才允许发散思维
                new_genes = extract_genes_from_result(tool_name, tool_output)
                if new_genes:
                    active_genes_bus = new_genes
                    print(f"     📥 [Path: {path_id}] [Bus] 捕获 {len(new_genes)} 个新基因")
            # === 修改结束 ===

            # 5. 保存证据 & 更新历史
            self._save_evidence_file(evidence_dir, i, tool_name, tool_args, tool_output, path_id)
            history.append({"step": tool_name, "args": tool_args, "result": tool_output})
            logs.append({"type":"step", "step": tool_name, "summary": {"step": tool_name, "type": "tool_result"}})
            
            # 6. 动态决策 (Post-Step)
            if self._handle_dynamic_decision(history, steps, i, logs, path_id, is_pre_step=False):
                if logs[-1].get("decision", {}).get("decision") == "STOP": break
            
            i += 1

        print(f"🏁 [Path: {path_id}] 执行完毕，综合结果中...")
        synthesis = self.synthesize_path(path_spec, history, task_json)
        # 传入 path_id 用于溯源
        synthesis = self._ensure_novelty_notes(synthesis, history, path_id)
        logs.append({"type":"synthesis", "path_id": path_id, "synthesis": synthesis})
        return {"path_id": path_id, "history": history, "synthesis": synthesis, "steps": steps}

    # === 私有辅助方法 ===

    def _inject_context_genes(self, tool_name, tool_args, active_genes_bus, path_id):
        #  KG/Literature 发现的基因可以被扔回 OmicsTool 进行全量数据验证
        target_tools = ["search_literature", "query_opentargets", "query_kg", "run_omics"]
        
        if tool_name in target_tools and active_genes_bus:
            existing = tool_args.get("genes") or tool_args.get("gene")
            # 如果参数为空、或者只是占位符/默认值，则注入总线中的基因
            if not existing or existing in ["<decide>", "TP53"]:
                print(f"     🔗 [Path: {path_id}] [Auto-Inject] 为 {tool_name} 注入 {len(active_genes_bus)} 个基因")
                tool_args["genes"] = active_genes_bus
                if "gene" in tool_args: del tool_args["gene"]

    def _save_evidence_file(self, directory, index, tool, args, result, path_id):
        try:
            fname = f"{directory}/step_{index+1}_{tool}_{datetime.now().strftime('%H%M%S')}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump({"path_id": path_id, "step": index, "tool": tool, "args": args, "result": result}, 
                          f, ensure_ascii=False, indent=2, cls=MongoDBJSONEncoder)
            print(f"     ✅ [Path: {path_id}] 证据已保存: {os.path.basename(fname)}")
        except Exception as e:
            print(f"     ⚠️ [Path: {path_id}] 保存失败: {e}")

    def _handle_dynamic_decision(self, history, steps, current_index, logs, path_id, is_pre_step=False) -> bool:
        current_step_name = ""
        if current_index < len(steps):
            item = steps[current_index]
            current_step_name = item.get("tool") if isinstance(item, dict) else item

        should_check = False
        if is_pre_step:
            if isinstance(current_step_name, str) and current_step_name.startswith("<"):
                should_check = True
        else:
            should_check = True

        if not should_check: return False

        decision = self.step_decide(history, self.available_tools)
        dec_type = decision.get("decision", "CONTINUE")
        logs.append({"type": "decide", "decision": decision})

        if dec_type == "STOP":
            print(f"🛑 [Path: {path_id}] 决策: 停止执行")
            return True
        elif dec_type == "INSERT":
            tool, args = decision.get("tool"), decision.get("args", {})
            if tool:
                steps.insert(current_index + 1, {"tool": tool, "args": args, "reason": "dynamic_insert"})
                print(f"     🔄 [Path: {path_id}] 动态插入: {tool}")
                if is_pre_step and current_step_name.startswith("<"):
                    if current_index < len(steps):
                        steps.pop(current_index)
                return True
        elif dec_type == "CONTINUE":
            if is_pre_step and current_step_name.startswith("<"):
                if current_index < len(steps):
                    steps.pop(current_index)
                return True
        return False

    def synthesize_path(self, path_spec: dict, intermediate: list, task_understanding: dict):
        cleaned_intermediate = []
        for item in intermediate:
            clean_item = item.copy()
            result = clean_item.get("result", {})
            tool_type = result.get("type", "")
            if isinstance(result, dict) and tool_type in ["search_literature", "search_pubmed_mongo", "query_mongo_local"]:
                summary_text = result.get("summary", "")
                clean_item["result"] = {"type": tool_type, "summary": summary_text}
            elif isinstance(result, dict) and result.get("type") == "run_omics":
                clean_item["result"] = {"type": "run_omics", "summary": "truncated"}
            cleaned_intermediate.append(clean_item)

        payload = {"path_spec": path_spec, "intermediate_outputs": cleaned_intermediate, "task_understanding": task_understanding}
        prompt = PATH_EXECUTOR_PROMPT.replace("{payload}", json.dumps(payload, ensure_ascii=False, indent=2, cls=MongoDBJSONEncoder))
        return self._llm(prompt)

    # === [证据链生成与过滤 ===
    def _ensure_novelty_notes(self, synthesis: dict, history: list, path_id: str):
        if not synthesis: return synthesis
        
        candidates = synthesis.get("candidate_targets", [])
        if not candidates: return synthesis
        
        gene_evidence_map = {}
        
        for idx, step_data in enumerate(history):
            step_num = idx + 1
            tool = step_data.get("step")
            result = step_data.get("result", {})
            
            # (A) Omics 证据 (兼容 Discovery 和 Verification 两种返回格式)
            omics_res = []
            if "results" in result:
                if isinstance(result["results"], dict): # Discovery Mode
                    omics_res = result["results"].get("top_upregulated", []) + result["results"].get("top_downregulated", [])
                elif isinstance(result["results"], list): # Verification Mode
                    omics_res = result["results"]

            for g_item in omics_res:
                # 兼容 gene_id (Discovery) 和 gene (Verification)
                g_name = g_item.get("gene_id") or g_item.get("gene")
                if g_name:
                    logfc = g_item.get("log2fc") or g_item.get("log2FoldChange")
                    padj = g_item.get("padj")
                    
                    if logfc is not None and padj is not None:
                         logfc_str = f"{logfc:.2f}"
                         padj_str = f"{padj:.1e}"
                         ev_str = f"[Step {step_num} Omics] logFC={logfc_str}, p={padj_str}"
                         if g_name.upper() not in gene_evidence_map: gene_evidence_map[g_name.upper()] = []
                         gene_evidence_map[g_name.upper()].append(ev_str)

            # (B) OpenTargets 证据
            if tool == "query_opentargets" and "results" in result:
                for rank, item in enumerate(result["results"]):
                    g_name = item.get("symbol")
                    if g_name:
                        score = item.get("score", 0)
                        ev_str = f"[Step {step_num} OpenTargets] Rank={rank+1}, Score={score:.2f}"
                        if g_name.upper() not in gene_evidence_map: gene_evidence_map[g_name.upper()] = []
                        gene_evidence_map[g_name.upper()].append(ev_str)
            
            # (C) 文献证据
            if tool in ["search_literature", "search_pubmed_mongo"]:
                
                # 🆕 优先检查是否存在结构化的 gene_details
                gene_details = result.get("gene_details", {})
                
                if gene_details:
                    # 如果有详情，直接精准匹配
                    for g_key, detail in gene_details.items():
                        count = detail.get("count", 0)
                        summary = detail.get("summary", "")
                        if count > 0:
                            # 格式：[Step X Lit] (5篇) 这是一个癌基因...
                            ev_str = f"[Step {step_num} Lit] ({count}篇) {summary}"
                            if g_key.upper() not in gene_evidence_map: gene_evidence_map[g_key.upper()] = []
                            gene_evidence_map[g_key.upper()].append(ev_str)
                
                else:
                    # ⚠️ 旧逻辑回退（如果没有 gene_details，才用总数）
                    n_res = result.get("n_results", 0)
                    if n_res > 0:
                        target_genes = step_data.get("args", {}).get("genes", [])
                        if isinstance(target_genes, str): target_genes = [target_genes]
                        single_gene = step_data.get("args", {}).get("gene")
                        if single_gene: target_genes.append(single_gene)

                        for tg in target_genes:
                            if not tg: continue
                            ev_str = f"[Step {step_num} Lit] 检索到 {n_res} 篇文献(总计)"
                            if tg.upper() not in gene_evidence_map: gene_evidence_map[tg.upper()] = []
                            gene_evidence_map[tg.upper()].append(ev_str)

            # (D) KG 证据
            if tool == "query_kg" and "evidence" in result:
                 # result["evidence"] 是一个 dict: {gene: [evidence_items]}
                 for gene_key, ev_list in result["evidence"].items():
                     if not ev_list: continue
                     # 汇总分数或条数
                     total_score = sum(e["score"] for e in ev_list)
                     sources = list(set(e["source"] for e in ev_list))
                     source_str = ",".join(sources)
                     ev_str = f"[Step {step_num} KG] {source_str} (Score={total_score:.1f})"
                     if gene_key.upper() not in gene_evidence_map: gene_evidence_map[gene_key.upper()] = []
                     gene_evidence_map[gene_key.upper()].append(ev_str)

        valid_candidates = []
        novelty_notes = {}
        
        for gene in candidates:
            gene_name = str(gene.get("gene") if isinstance(gene, dict) else gene)
            gene_upper = gene_name.upper()
            
            evidences = gene_evidence_map.get(gene_upper, [])
            if not evidences: continue 
            
            is_ot_known = any("OpenTargets" in e and "Rank=" in e and int(e.split("Rank=")[1].split(",")[0]) <= 50 for e in evidences)
            is_novel = not is_ot_known
            
            reason_str = f"[{path_id}] " + " | ".join(evidences)
            
            novelty_notes[gene_name] = {
                "novel": is_novel,
                "reason": reason_str
            }
            valid_candidates.append(gene)
            
        def evidence_score(g):
            g_name = str(g.get("gene") if isinstance(g, dict) else g)
            return len(novelty_notes.get(g_name, {}).get("reason", "").split("|"))
            
        valid_candidates.sort(key=evidence_score, reverse=True)

        synthesis["candidate_targets"] = valid_candidates
        synthesis["novelty_notes"] = novelty_notes
        return synthesis