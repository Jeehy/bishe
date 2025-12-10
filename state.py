# targets/state.py
import operator
from typing import List, Dict, Any, Annotated, TypedDict

def merge_lists(a: list, b: list) -> list:
    """合并两个列表（用于并行执行时的日志聚合）"""
    return (a or []) + (b or [])

class TargetDiscoveryState(TypedDict):
    """
    全局状态对象，存储从任务理解到最终反思的所有数据。
    """
    # --- 输入 ---
    user_input: str
    
    # --- 规划阶段 ---
    task_understanding: Dict[str, Any]      # 任务理解结果
    planned_paths: List[Dict[str, Any]]     # 规划的路径列表
    
    # --- 执行阶段 ---
    path_results: Annotated[List[Dict[str, Any]], merge_lists] 
    logs: Annotated[List[Dict[str, Any]], merge_lists]
    
    # --- 综合与反思 ---
    final_candidates: List[Dict[str, Any]]  # 去重后的最终候选靶点
    reflection: Dict[str, Any]              # 最终反思结果
    
    # --- 学习阶段 (ACE Curator) ---
    playbook_updates: List[Dict[str, Any]]  # 本次运行提取的新策略