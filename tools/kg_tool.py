# tools/kg_tool.py
import pandas as pd
from py2neo import Graph

class KGTool:
    """
    综合 KG 工具 (Hetionet版 - 中文解释增强版)
    功能：
    1. 多模态挖掘 (PPI, Pathway, Anatomy)
    2. 生成中文推理依据
    3. 输出汇总列表 + 分类明细
    """

    def __init__(self): 
        # 使用 Hetionet 公共数据库 
        self.uri = "bolt://neo4j.het.io:7687" 
        self.user = "neo4j" 
        self.password = "neo4j" 
        try: 
            self.graph = Graph(self.uri, auth=(self.user, self.password)) 
            print("KGTool: 成功连接至 Hetionet 公共数据库") 
        except Exception as e: 
            print(f"KGTool: 连接失败 - {e}") 
            self.graph = None

    # 已知关联
    def _query_known_targets(self, disease_name):
        cypher = """
        MATCH (d:Disease)-[r:ASSOCIATES_DaG]-(g:Gene)
        WHERE toLower(d.name) = toLower($name)
        RETURN g.name AS gene
        """
        data = self.graph.run(cypher, name=disease_name).data()
        results = []
        for row in data:
            results.append({
                "gene": row["gene"],
                "source": "Known Association",
                "detail": "已有文献/数据库支持的验证靶点",
                "score": 100.0,
                "is_known": True
            })
        return results

    # PPI 推断  
    def _discover_by_ppi(self, disease_name, limit=15):
        cypher = """
        MATCH (d:Disease)-[:ASSOCIATES_DaG]-(seed:Gene)
        WHERE toLower(d.name) = toLower($name)
        MATCH (seed)-[:INTERACTS_GiG]-(candidate:Gene)
        WHERE NOT (d)-[:ASSOCIATES_DaG]-(candidate)
        RETURN candidate.name AS gene, 
               count(DISTINCT seed) AS raw_count, 
               collect(DISTINCT seed.name)[0..3] AS evidence_list
        ORDER BY raw_count DESC LIMIT $limit
        """
        data = self.graph.run(cypher, name=disease_name, limit=limit).data()
        results = []
        for row in data:
            evidence_str = ", ".join(row["evidence_list"])
            results.append({
                "gene": row["gene"],
                "source": "Inference (PPI)",
                "detail": f"与 {row['raw_count']} 个已知基因直接相互作用（如: {evidence_str}）",
                "score": row["raw_count"] * 2.0,
                "is_known": False
            })
        return results

    #  Pathway 推断
    def _discover_by_pathway(self, disease_name, limit=15):
        cypher = """
        MATCH (d:Disease)-[:ASSOCIATES_DaG]-(seed:Gene)-[:PARTICIPATES_GpPW]->(p:Pathway)
        WHERE toLower(d.name) = toLower($name)
        MATCH (p)<-[:PARTICIPATES_GpPW]-(candidate:Gene)
        WHERE NOT (d)-[:ASSOCIATES_DaG]-(candidate)
        RETURN candidate.name AS gene, 
               count(DISTINCT p) AS raw_count, 
               collect(DISTINCT p.name)[0..2] AS evidence_list
        ORDER BY raw_count DESC LIMIT $limit
        """
        data = self.graph.run(cypher, name=disease_name, limit=limit).data()
        results = []
        for row in data:
            evidence_str = ", ".join(row["evidence_list"])
            results.append({
                "gene": row["gene"],
                "source": "Inference (Pathway)",
                "detail": f"与已知基因共同参与 {row['raw_count']} 条通路（如: {evidence_str}）",
                "score": row["raw_count"] * 1.0,
                "is_known": False
            })
        return results

    # 多跳推断
    def _discover_by_anatomy(self, disease_name, limit=15):
        cypher = """
        MATCH (d:Disease)-[:LOCALIZES_DlA]-(a:Anatomy)-[:EXPRESSES_AeG]-(candidate:Gene)
        WHERE toLower(d.name) = toLower($name)
        AND NOT (d)-[:ASSOCIATES_DaG]-(candidate)
        RETURN candidate.name AS gene, 
               count(DISTINCT a) AS raw_count, 
               collect(DISTINCT a.name)[0..2] AS evidence_list
        ORDER BY raw_count DESC LIMIT $limit
        """
        data = self.graph.run(cypher, name=disease_name, limit=limit).data()
        results = []
        for row in data:
            evidence_str = ", ".join(row["evidence_list"])
            results.append({
                "gene": row["gene"],
                "source": "Inference (Anatomy)",
                "detail": f"基因在疾病相关组织中高表达（如: {evidence_str}）",
                "score": row["raw_count"] * 0.5,
                "is_known": False
            })
        return results

    def run(self, context=None):
        if self.graph is None:
            return {"error": "Neo4j connection failed"}

        target_disease = context.get("disease", "liver cancer") if context else "liver cancer"
        print(f"\nKGTool: 开始查找 [{target_disease}] ...\n")
        known_list = self._query_known_targets(target_disease)
        ppi_list = self._discover_by_ppi(target_disease)
        pathway_list = self._discover_by_pathway(target_disease)
        anatomy_list = self._discover_by_anatomy(target_disease)

        evidence_map = {}
        all_genes = set()
        def add_items(items):
            for it in items:
                gene = it["gene"]
                all_genes.add(gene)
                evidence_map.setdefault(gene, []).append(it)

        add_items(known_list)
        add_items(ppi_list)
        add_items(pathway_list)
        add_items(anatomy_list)

        known_genes = {item["gene"] for item in known_list}
        novel_genes = []
        for gene in all_genes:
            evidence_list = evidence_map[gene]
            # 只要某来源标记了 is_known=True，这个基因就排除
            if any(item.get("is_known", False) for item in evidence_list):
                continue
            novel_genes.append(gene)
        def total_score(gene):
            return sum(item["score"] for item in evidence_map[gene])
        sorted_novel_targets = sorted(novel_genes, key=total_score, reverse=True)
        res = res = {
            "type": "query_kg",
            "disease": target_disease,
            # 去除 Known
            "linked_targets": sorted_novel_targets,
            # B：每个基因有多条证据（简短理由 + 分数）
            "evidence": {
                g: evidence_map[g] for g in sorted_novel_targets
            },
            "description": f"共发现 {len(sorted_novel_targets)} 个潜在靶点（已剔除所有已知靶点）"
        }

        return res


# --- 运行测试 ----
if __name__ == "__main__":
    tool = KGTool()
    tool.run()
