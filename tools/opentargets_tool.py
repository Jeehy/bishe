import requests

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
        disease = context.get("topic", "hepatocellular carcinoma")

        EFO_MAP = {
            "hepatocellular carcinoma": "EFO_0000186",
            "liver cancer": "EFO_0000186",
            "hcc": "EFO_0000186"
        }

        efo = EFO_MAP.get(disease.lower())
        if not efo:
            return {"type":"query_opentargets","results":[],"error":"no efo id"}

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
            return {"type":"query_opentargets","results":[],"error":data["error"]}

        try:
            rows = data["data"]["disease"]["associatedTargets"]["rows"]
            res = [{
                "symbol": r["target"]["approvedSymbol"],
                "name": r["target"]["approvedName"],
                "score": r["score"]
            } for r in rows]
        except:
            res = []

        return {
            "type": "query_opentargets",
            "results": res,
            "n_results": len(res)
        }
