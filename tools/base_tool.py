# tools/base_tool.py
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """工具基类：所有工具实现需继承并实现 run(self, input_data: dict) -> dict"""
    @abstractmethod
    def run(self, input_data: dict) -> dict:
        raise NotImplementedError
