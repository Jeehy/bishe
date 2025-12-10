# tools/tool_executor.py

from tools.kg_tool import KGTool
from tools.omics_tool import OmicsTool
from tools.mongo_local_tool import MongoLocalTool
from tools.opentargets_tool import OpenTargetsTool

# ================================
# 可用工具
# ================================
def get_available_tools():
    return [
        "query_kg",
        "run_omics",
        "query_opentargets",
        "search_literature"
    ]


class ToolExecutor:
    def __init__(self):
        # 初始化所有真实工具
        self.kg = KGTool()
        self.omics = OmicsTool()
        self.mongo_local = MongoLocalTool()
        self.opentargets = OpenTargetsTool()

        # 工具映射
        self.tools = {
            "query_kg": self._run_kg,
            "run_omics": self._run_omics,
            "query_opentargets": self._run_opentargets,
            "search_literature": self._run_mongo_local,
            "query_mongo_local": self._run_mongo_local
        }

    # ==================================
    # 工具调用
    # ==================================
    def _run_kg(self, params):
        return self.kg.run(params)

    def _run_omics(self, params):
        return self.omics.run(params)

    def _run_mongo_local(self, params):
        return self.mongo_local.run(params)

    def _run_opentargets(self, params):
        return self.opentargets.run(params)

    # ==================================
    # 执行入口
    # ==================================
    def execute(self, step, task_context, history=None):
        func = self.tools.get(step)
        if func is None:
            return {"type": step, "error": f"未知工具: {step}"}
        
        for k, v in task_context.items():
            if isinstance(v, str) and "<decide>" in v:
                msg = f"Parameter '{k}' is a placeholder '{v}'. Execution skipped."
                print(f"   ⚠️ [Executor] {msg} Waiting for Decider to fill it.")
                return {
                    "type": step,
                    "status": "skipped",
                    "reason": msg,
                    "summary": "Step skipped due to pending decision (<decide> placeholder)."
                }
            
        # 1. 基础 params 包含 task
        params = {"task": task_context.get("task", {})}
        
        # 2. 将 task_context 中的其他所有参数（如 genes, query）合并进来
        for k, v in task_context.items():
            if k != "task":
                params[k] = v
                
        # 3. 注入历史
        if history is not None:
            params["history"] = history
            
        return func(params)
