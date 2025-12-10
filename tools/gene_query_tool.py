# tools/gene_tool.py
import pandas as pd
import os

class GeneTool:
    """
    通过 DESeq2 差异表达文件查询单基因
    """

    def __init__(self, de_path="D:/Bit/bishe/data/DESeq2_results_all.csv"):
        self.de_path = de_path
        self.df = pd.read_csv(de_path) if os.path.exists(de_path) else None

    def run(self, context):
        gene = context.get("gene")
        if not gene:
            return {"type":"query_gene","error":"no gene"}

        if self.df is None:
            return {"type":"query_gene","error":"DESeq2 文件未加载"}

        row = self.df[self.df["gene_id"] == gene]
        if row.empty:
            return {
                "type":"query_gene",
                "gene":gene,
                "found":False,
                "error":"基因不在差异分析中"
            }

        r = row.iloc[0]

        log2fc = float(r["log2FoldChange"])
        padj = float(r["padj"])

        return {
            "type": "query_gene",
            "result": {
                "gene": gene,
                "log2FoldChange": log2fc,
                "padj": padj,
                "direction": "up" if log2fc > 0 else "down",
                "significant": padj < 0.05
            }
        }
