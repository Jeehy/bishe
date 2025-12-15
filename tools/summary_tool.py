# deepseek_api.py
# DeepSeek 调用封装（支持真实 API 与本地 stub）
# - 若设置环境变量 DEEPSEEK_URL/DEEPSEEK_API_KEY，则调用真实接口
# - 否则使用本地 deterministic stub 以便离线开发
from dotenv import load_dotenv
import os
import json
import requests

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def model_call_real(prompt: str, model_name: str = "deepseek-chat", timeout: int = 60) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    conversation = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你是严谨的科研助理，请尽量只输出 JSON。"},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    
    # 增加超时设置和错误打印
    response = requests.post(DEEPSEEK_URL, headers=headers, json=conversation, timeout=timeout)
    
    if response.status_code != 200:
        print(f"\n[DeepSeek API Error] Status: {response.status_code}")
        print(f"[Error Body]: {response.text}") # 关键：打印出服务端返回的具体错误信息
        response.raise_for_status()

    j = response.json()
    return j["choices"][0]["message"]["content"]

def model_call_stub(prompt: str, model_name: str = "deepseek-chat") -> str:
    """
        简单 deterministic stub，用于离线开发和测试
    """
    p = prompt.lower()
    # task understanding
    if "任务理解" in p or "task understanding" in p:
        return json.dumps({
            "topic": "liver cancer",
            "goal": "target_discovery",
            "known_databases": ["OpenTargets", "PubMed"],
            "suggested_start": ["query_opentargets", "run_omics"],
            "reason": "基于疾病名优先查询已知靶点，再验证组学"
        }, ensure_ascii=False)
    # path planner
    if "路径生成" in p or "path planner" in p or "generate" in p and "paths" in p:
        return json.dumps([
            {"path_id": "p1", "steps": ["query_opentargets", "run_omics", "search_pubmed"], "reason": "先查已知靶点再验证表达"},
            {"path_id": "p2", "steps": ["run_omics", "query_kg", "search_pubmed"], "reason": "从差异表达出发，匹配图谱与文献"}
        ], ensure_ascii=False)
    # executor synthesis
    if "推理综合" in p or "synthesize" in p or "reasoning_chain" in p:
        return json.dumps({
            "reasoning_chain": [
                "OpenTargets 显示 TP53 与肝癌相关",
                "差异分析中 RPS6KA1 上调显著",
                "文献检索显示 TP53 在肝癌中多次被报道"
            ],
            "candidate_targets": ["TP53", "RPS6KA1"],
            "confidence": 0.88,
            "new_queries": ["查询 TP53 在 OpenTargets 中的关联证据"],
            "change_path": False
        }, ensure_ascii=False)
    # reflector
    if "反思" in p or "reflector" in p or "reflect" in p:
        return json.dumps({
            "consensus": ["TP53"],
            "converged": True,
            "suggested_paths": [],
            "new_queries": []
        }, ensure_ascii=False)
    return json.dumps({"note": "stub fallback"}, ensure_ascii=False)

def summary(prompt: str, model_name: str = "deepseek-chat") -> str:
    """统一入口：优先调用真实 API，失败或未配置则使用 stub"""
    if DEEPSEEK_URL and DEEPSEEK_API_KEY:
        try:
            return model_call_real(prompt, model_name=model_name)
        except Exception as e:
            print("DeepSeek API 调用失败，回退到本地 stub:", e)
            return model_call_stub(prompt, model_name=model_name)
    else:
        return model_call_stub(prompt, model_name=model_name)
