#!/usr/bin/env python3
# main.py
import sys
import json
from datetime import datetime
from graph_system import GraphTargetDiscovery

def main():
    # 1. 获取输入
    if len(sys.argv) > 1:
        task_input = " ".join(sys.argv[1:])
    else:
        task_input = input("请输入任务（例如：尝试发现肝癌潜在靶点）：\n").strip()
    
    # 2. 初始化图系统
    app = GraphTargetDiscovery()
    
    # 3. 运行图 (Invoke)
    print(f"\n=== 启动 Target Discovery Agent (Graph Mode) ===")
    initial_state = {"user_input": task_input}
    
    try:
        # 运行 LangGraph
        final_state = app.graph.invoke(initial_state)
        
        # 4. 输出结果
        print("\n=== LLM 分析结果 ===")
        finals = final_state.get("final_candidates", [])
        task_info = final_state.get("task_understanding", {})
        task_type = task_info.get("task_type", "discovery") # 默认为发现模式
        if not finals:
            print("未发现候选")
        else:
            for f in finals:
                gene = f.get("gene")
                novel = f.get("novel")
                reason = f.get("reason", "")
                tag = "新颖" if novel else "已知"

                if task_type == "verification":
                    if reason:
                        print(f"- 验证 {gene}")
                        # 按照 planner_system 中定义的 " | " 分隔符拆分
                        evidences = reason.split(" | ")
                        for ev in evidences:
                            # 去掉可能的首尾空格
                            print(f"  * {ev.strip()}")
                else:
                    print(f"- {gene} ({tag}) | {reason}...")

        # 5. 保存完整日志
        out = {
            "task": task_input,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "result": {
                "final_candidates": finals,
                "reflection": final_state.get("reflection"),
                "playbook_updates": final_state.get("playbook_updates")
            },
            "logs": final_state.get("logs", [])
        }
        
        with open("run_log.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"\n详细记录已保存到 run_log.json")
        
    except Exception as e:
        print(f"\n❌ 运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()