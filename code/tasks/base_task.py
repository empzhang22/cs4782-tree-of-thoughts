from abc import ABC, abstractmethod

from tot.node import ThoughtNode


class BaseTask(ABC):
    @abstractmethod
    def get_input(self, idx: int) -> str:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def is_success(self, node: ThoughtNode) -> bool:
        ...

    @abstractmethod
    def make_root(self, idx: int) -> ThoughtNode:
        ...
