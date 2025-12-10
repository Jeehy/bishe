# targets/playbook.py
import os
import json
import time
import uuid
from typing import List, Dict

BASE_DIR = os.path.dirname(__file__)
PLAYBOOK_PATH = os.path.join(BASE_DIR, "hcc_playbook.json")

class Playbook:
    def __init__(self):
        self.strategies = self._load_strategies()

    def _load_strategies(self) -> List[Dict]:
        """加载现有的策略库"""
        if os.path.exists(PLAYBOOK_PATH):
            try:
                with open(PLAYBOOK_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data.get("strategies", [])
            except Exception:
                return []
        return []

    def save(self):
        """持久化保存"""
        data = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "strategies": self.strategies
        }
        with open(PLAYBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # bishe/playbook.py

    def add_strategy(self, trace_data: Dict):
        """
        添加一条具体的执行案例
        trace_data 结构示例:
        {
            "task": "发现肝癌靶点",
            "status": "success",  # 或 "failure"
            "steps_summary": ["query_opentargets", "run_omics"],
            "step_details": [
                {"step": "query_opentargets", "effective": True, "note": "返回 50 条数据"},
                {"step": "run_omics", "effective": False, "note": "无显著差异基因"}
            ],
            "conclusion": "数据源缺失导致路径中断"
        }
        """
        # 简单指纹去重（防止完全一样的运行记录重复存）
        import hashlib
        # 这里用 task + status + steps 做指纹
        fingerprint = f"{trace_data.get('task')}-{trace_data.get('status')}-{str(trace_data.get('steps_summary'))}"
        fingerprint_hash = hashlib.md5(fingerprint.encode()).hexdigest()

        for s in self.strategies:
            if s.get("fingerprint") == fingerprint_hash:
                return

        entry = {
            "id": str(uuid.uuid4())[:8],
            "type": "execution_trace", # 标记这是具体案例
            "fingerprint": fingerprint_hash,
            "data": trace_data,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.strategies.append(entry)
        self.save()

    def retrieve_strategies(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索相关策略 
        """
        if not self.strategies:
            return []
        scored = []
        query_lower = query.lower()
        
        for s in self.strategies:
            score = 0
            
            # 提取出一个 content_text 用于匹配
            content_text = ""
            category_text = ""
            
            if "content" in s:
                # Case A: 直接读取 content
                content_text = s["content"]
                category_text = s.get("category", "")
            elif "data" in s:
                # Case B: 从 data 中提取 task 和 conclusion 组合成文本
                trace = s["data"]
                # 组合任务名和结论，以便能搜到
                task_str = trace.get("task", "")
                concl_str = trace.get("conclusion", "")
                content_text = f"{task_str} {concl_str}"
                category_text = "execution_trace"
            else:
                continue

            content_lower = content_text.lower()
            category_lower = category_text.lower()

            # === 2. 打分逻辑 ===
            if query_lower in content_lower or content_lower in query_lower:
                score += 10
            if category_lower in query_lower:
                score += 5
            
            if len(query_lower) >= 2:
                for i in range(len(query_lower) - 1):
                    sub = query_lower[i:i+2]
                    if sub in content_lower:
                        score += 1
            
            if score > 0:
                scored.append((score, s))
        
        # 按分数降序
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored[:top_k]]
        
        # 自动回退机制
        if not results and self.strategies:
            return self.strategies[-top_k:]
            
        return results