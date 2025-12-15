# targets/graph_system.py
import re
import concurrent.futures
from typing import Dict
from langgraph.graph import StateGraph, END
from planner_system import PlannerSystem
from playbook import Playbook
from state import TargetDiscoveryState

class GraphTargetDiscovery:
    def __init__(self):
        self.core_system = PlannerSystem()
        self.playbook = Playbook()
        
        # 初始化图
        self.graph = self.build()

    def build(self) -> StateGraph:
        workflow = StateGraph(TargetDiscoveryState)

        # === 1. 定义节点 (Nodes) ===
        workflow.add_node("planner", self.planner)
        workflow.add_node("executor", self.executor)
        workflow.add_node("synthesizer", self.synthesizer)
        workflow.add_node("curator", self.curator)

        # === 2. 定义流程 (Edges) ===
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "synthesizer")
        workflow.add_edge("synthesizer", "curator")
        workflow.add_edge("curator", END)

        return workflow.compile()

    # --- Node: 规划 (引入 Playbook ) ---
    def planner(self, state: TargetDiscoveryState) -> Dict:
        user_input = state["user_input"].strip()
        print(f"🔒 [Planner] 收到任务: {user_input}")
        
        paths = []
        task_info = {}
        # === 规则 1: 验证模式 (格式: "验证" + 基因名) ===
        # 正则解释:
        # ^       : 从字符串开头匹配
        # 验证    : 必须包含“验证”二字
        # \s* : 允许中间有空格，也可以没有 (兼容 "验证TP53" 和 "验证 TP53")
        # ([a-zA-Z0-9]+) : 捕获组，提取后面的英文/数字作为基因名
        match = re.match(r"^验证\s*([a-zA-Z0-9]+)", user_input)
        if match:
            # 提取基因名并转大写
            target_gene = match.group(1).upper()
            print(f"   🎯 [规则命中] 验证模式 | 目标基因: {target_gene}")

            # 构造验证任务 (无需 LLM)
            task_info = {
                "task_type": "verification",
                "target_gene": target_gene,
                "context": "Hepatocellular Carcinoma"
            }

            # 构造验证路径: OpenTargets -> Literature -> Omics
            paths = [{
                "path_id": f"verify_{target_gene}",
                "steps": [
                    {
                        "tool": "query_opentargets", 
                        "args": {"genes": [target_gene]}
                    },
                    {
                        "tool": "search_literature", 
                        "args": {"genes": [target_gene]}
                    },
                    {
                        "tool": "run_omics", 
                        "args": {"genes": [target_gene]} 
                    }
                ]
            }]

        # === 规则 2: 发现模式 (其他所有输入) ===
        else:
            print(f"   🔍 [默认模式] 发现模式 (Discovery Mode)")
            
            # 调用 LLM 理解复杂任务
            task_info = self.core_system.understand_task(user_input)
            
            # 构造发现路径
            paths = [{
                "path_id": "discovery_pipeline",
                "steps": [
                    {
                        "tool": "run_omics", 
                        "args": {} 
                    },
                    {
                        "tool": "query_kg", 
                        "args": {"genes": "<decide>"} 
                    },
                    {
                        "tool": "search_literature", 
                        "args": {"genes": "<decide>"} 
                    }
                ]
            }]

        print(f"   ✅ 路径规划完成")

        return {
            "task_understanding": task_info,
            "planned_paths": paths,
            "logs": [{"type": "plan", "content": paths}]
        }
    

    # --- Node: 执行 (并行加速) ---
    def executor(self, state: TargetDiscoveryState) -> Dict:
        paths = state["planned_paths"]
        task = state["task_understanding"]
        print(f"🚀 [Executor] 启动并行执行，共 {len(paths)} 条路径...")

        path_results = []
        logs = []
        if not paths:
            print("   ⚠️ 没有路径需要执行")
            return {"path_results": [], "logs": []}
        # 定义单个路径的运行函数
        def run_single_path(path_spec):
            # 每个线程保留独立的 log list
            local_logs = []
            try:
                res = self.core_system.execute_path_with_reflection(path_spec, task, local_logs)
                return res, local_logs
            except Exception as e:
                err_msg = f"Path {path_spec.get('path_id')} failed: {str(e)}"
                print(f"   ❌ {err_msg}")
                return {"error": err_msg}, local_logs

        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 提交任务
            future_to_path = {executor.submit(run_single_path, p): p for p in paths}
            
            for future in concurrent.futures.as_completed(future_to_path):
                try:
                    res, l_logs = future.result()
                    path_results.append(res)
                    logs.extend(l_logs)
                except Exception as e:
                    print(f"   ❌ 线程执行异常: {e}")

        return {
            "path_results": path_results,
            "logs": logs
        }

    # --- Node: 综合 (去重与反思) ---
    def synthesizer(self, state: TargetDiscoveryState) -> Dict:
        print("🧠 [Synthesizer] 正在综合结果...")
        results = state["path_results"]
        task_info = state.get("task_understanding", {}) # 获取任务信息

        reflection = self.core_system.reflect_paths(results)
        final_candidates = self.deDuplicate(results)
        
        # === 🆕 [Fix] 验证模式强制过滤 ===
        # 如果是验证任务，只保留用户指定的目标基因，剔除 OpenTargets 等工具带来的"伴随"结果
        if task_info.get("task_type") == "verification":
            target_gene = task_info.get("target_gene", "").upper()
            if target_gene:
                print(f"   🔒 [Verification Filter] 验证模式生效，仅保留目标基因: {target_gene}")
                filtered = []
                for cand in final_candidates:
                    # 获取候选基因名 (兼容字典或字符串格式)
                    c_gene = cand.get("gene") if isinstance(cand, dict) else str(cand)
                    if str(c_gene).upper() == target_gene:
                        filtered.append(cand)
                
                final_candidates = filtered
                
                # 如果过滤后为空（可能是别名问题或没查到），做一个兜底提示
                if not final_candidates:
                    print(f"   ⚠️ 警告：目标基因 {target_gene} 未出现在候选列表中，可能缺乏证据。")

        return {
            "reflection": reflection,
            "final_candidates": final_candidates
        }

    # --- Node: 策展 (ACE Curator 学习机制) ---
    def curator(self, state: TargetDiscoveryState) -> Dict:
        print("📚 [Curator] 正在复盘并记录执行细节...")
        task_input = state["user_input"]
        path_results = state["path_results"]
        
        new_strategies_count = 0
        
        for path_res in path_results:
            # 1. 提取基本信息
            path_id = path_res.get("path_id", "unknown")
            steps_executed = path_res.get("steps", [])
            history = path_res.get("history", [])
            synthesis = path_res.get("synthesis", {})
            error = path_res.get("error")
            
            # 分析每一步的有效性 (Step Effectiveness)
            step_details = []
            for h in history:
                step_name = h.get("step")
                result = h.get("result", {})
                
                is_effective = True
                note = "执行正常"
                if isinstance(result, dict):
                    if result.get("error"):
                        is_effective = False
                        note = f"错误: {result.get('error')}"
                    elif "n_results" in result and result["n_results"] == 0:
                        is_effective = False
                        note = "无数据返回"
                    elif "n_significant" in result and result["n_significant"] == 0:
                        is_effective = False
                        note = "无显著结果"
                step_details.append({
                    "step": step_name,
                    "effective": is_effective,
                    "note": note
                })
            status = "success"
            conclusion = "成功发现候选"
            candidates = synthesis.get("candidate_targets", [])
            if error:
                status = "failure"
                conclusion = f"运行报错: {error}"
            elif not candidates:
                status = "failure" # 或者 "partial_success"
                conclusion = "路径跑通但未发现有价值靶点"
            
            # 4. 构建 Trace 数据
            trace_data = {
                "task": task_input,
                "status": status,
                "steps_summary": steps_executed,
                "step_details": step_details,
                "conclusion": conclusion
            }
            
            # 5. 保存到 Playbook
            self.playbook.add_strategy(trace_data)
            new_strategies_count += 1

        print(f"   ✅ 已记录 {new_strategies_count} 条执行案例 (含成功与失败)")
        return {"playbook_updates": []} # 这里可以返回空，因为已经直接操作了 self.playbook

    # --- 去重 ---
    def deDuplicate(self, paths_results):
        candidates_map = {}
        if not paths_results:
            return []
        for pr in paths_results:
            if "error" in pr: continue
            
            syn = pr.get("synthesis") or {}
            cands = syn.get("candidate_targets") or syn.get("candidates") or []
            novnotes = syn.get("novelty_notes") or {}
            
            for c in cands:
                if isinstance(c, dict):
                    gene_raw = c.get("gene") or c.get("symbol")
                else:
                    gene_raw = str(c)
                
                if not gene_raw: continue
                gene_key = str(gene_raw).upper()
                entry = {"gene": gene_raw, "novel": True, "reason": ""}
                # 获取 Novelty 信息
                if gene_raw in novnotes:
                    val = novnotes[gene_raw]
                    if isinstance(val, dict):
                        entry.update(val)
                    else:
                        entry["novel"] = bool(val)
                if gene_key in candidates_map:
                    # 如果已存在，且新结果说它是已知的(novel=False)，则覆盖
                    if candidates_map[gene_key].get("novel") and not entry["novel"]:
                        candidates_map[gene_key] = entry
                else:
                    candidates_map[gene_key] = entry        
        return list(candidates_map.values())