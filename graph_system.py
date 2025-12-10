# targets/graph_system.py
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
        user_input = state["user_input"]
        print(f"🔍 [Planner] 正在分析任务: {user_input[:50]}...")

        # 1. 检索历史策略
        # 注意：这里 retrieve_strategies 内部可能需要简单适配一下，
        # 如果你的检索是纯关键词匹配，现有的逻辑应该也能搜到（因为 'data' 字段里有 task 文本）
        strategies = self.playbook.retrieve_strategies(user_input, top_k=3)
        strategy_context = ""
        if strategies:
            print(f"   📖 检索到 {len(strategies)} 条相关历史案例")
            formatted_cases = []
            for i, s in enumerate(strategies):
                data = s.get("data", {})
                if not data: continue 
                
                # 将结构化数据转为自然语言描述
                status_icon = "✅" if data.get("status") == "success" else "❌"
                raw_steps = data.get("steps_summary", [])
                safe_steps = []
                for st in raw_steps:
                    if isinstance(st, dict):
                        # 如果是字典（带参数的步骤），只提取工具名
                        tool_name = st.get("tool", str(st))
                        # 可选：如果你想让 Prompt 看到参数，可以写成 f"{tool_name}({st.get('args')})"
                        # 这里为了简洁，只用工具名
                        safe_steps.append(tool_name)
                    else:
                        safe_steps.append(str(st))
                
                steps_str = " -> ".join(safe_steps)
                
                # 提取关键的失败点或亮点
                details_str = ""
                for step in data.get("step_details", []):
                    if not step["effective"]:
                        details_str += f"\n      - ⚠️ 步骤 [{step['step']}] 效果不佳: {step['note']}"
                
                case_desc = (
                    f"案例 {i+1} [{status_icon} {data.get('status')}]:\n"
                    f"    路径: {steps_str}\n"
                    f"    结果: {data.get('conclusion')}"
                    f"{details_str}"
                )
                formatted_cases.append(case_desc)
            strategy_context = "\n【历史执行经验参考】:\n" + "\n".join(formatted_cases)

        # 2. 理解任务 (注入策略上下文)
        enhanced_input = f"{user_input}\n{strategy_context}"
        task = self.core_system.understand_task(enhanced_input)
        # 3. 规划路径
        planned_resp = self.core_system.plan_paths(task)
        
        # 兼容 List 和 Dict 两种返回格式 ===
        if isinstance(planned_resp, list):
            paths = planned_resp
        elif isinstance(planned_resp, dict):
            paths = planned_resp.get("paths", [])
        else:
            print(f"⚠️ [Planner] 警告：无法解析 LLM 返回的路径格式: {type(planned_resp)}")
            paths = []
            
        print(f"   ✅ 规划了 {len(paths)} 条路径")

        return {
            "task_understanding": task,
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
        reflection = self.core_system.reflect_paths(results)
        final_candidates = self.deDuplicate(results)
        
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
                
                # 简单判断有效性规则：
                # - 报错了 -> 无效
                # - 返回结果数量为0 -> 无效 (针对查询类工具)
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