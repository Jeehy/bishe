import requests
import json

class OpenTargetsTool:
    BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    def __init__(self):
        pass

    def _run_query(self, query, variables=None):
        try:
            response = requests.post(
                self.BASE_URL,
                json={"query": query, "variables": variables},
                timeout=20
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def run(self, context):
        """
        context 参数说明:
        - topic: 疾病名称 (默认 hepatocellular carcinoma)
        - genes: [可选] 基因列表 (例如 ["TP53"])。如果存在，则进入验证模式，仅筛选这些基因。
        """
        disease = context.get("topic", "hepatocellular carcinoma")
        
        # 1. 获取目标基因列表 (支持直接传列表或嵌套在 args 中)
        target_genes = context.get("genes") or context.get("args", {}).get("genes", [])
        # 转为大写集合以便匹配
        target_genes_set = set(g.upper() for g in target_genes) if target_genes else None

        EFO_MAP = {
            "hepatocellular carcinoma": "EFO_0000186",
            "liver cancer": "EFO_0000186",
            "hcc": "EFO_0000186"
        }

        efo = EFO_MAP.get(disease.lower())
        if not efo:
            return {"type":"query_opentargets", "results":[], "error":f"No EFO ID found for {disease}"}

        # 2. 查询该疾病关联的前 200 个靶点 (按分数排序)
        query = """
        query diseaseTargets($efo_id: String!) {
          disease(efoId: $efo_id) {
            associatedTargets(page: {index: 0, size: 200}) {
              rows {
                target {
                  approvedSymbol
                  approvedName
                }
                score
              }
            }
          }
        }
        """

        data = self._run_query(query, {"efo_id": efo})
        if "error" in data:
            return {"type":"query_opentargets", "results":[], "error":data["error"]}

        try:
            rows = data["data"]["disease"]["associatedTargets"]["rows"]
            res = [{
                "symbol": r["target"]["approvedSymbol"],
                "name": r["target"]["approvedName"],
                "score": round(r["score"], 4) # 保留4位小数
            } for r in rows]
        except Exception as e:
            return {"type":"query_opentargets", "results":[], "error":f"Parsing error: {str(e)}"}

        # 3. [关键修改] 验证模式过滤
        # 如果指定了 genes，只返回这些基因的结果
        if target_genes_set:
            print(f"    🔍 [OpenTargets] 正在筛选特定基因: {target_genes_set}")
            filtered_res = [r for r in res if r["symbol"].upper() in target_genes_set]
            
            # 检查是否有基因没找到
            found_symbols = set(r["symbol"].upper() for r in filtered_res)
            missing_genes = target_genes_set - found_symbols
            
            # 为未找到的基因添加空记录 (让 Agent 知道没数据)
            for missing in missing_genes:
                filtered_res.append({
                    "symbol": missing,
                    "name": "Unknown or Not in Top 200",
                    "score": 0.0,
                    "status": "Not Found"
                })
                
            return {
                "type": "query_opentargets_verification",
                "results": filtered_res,
                "n_results": len(filtered_res)
            }

        # 4. 发现模式 (返回 Top 列表)
        return {
            "type": "query_opentargets_discovery",
            "results": res, # 返回 Top 200
            "n_results": len(res)
        }