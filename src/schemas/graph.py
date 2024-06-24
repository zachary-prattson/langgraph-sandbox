import operator

from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage


class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
