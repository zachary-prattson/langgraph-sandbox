from langchain_core.messages import HumanMessage

from utils import set_env
from workflow.graph import super_graph


if __name__ == "__main__":
    set_env("OPENAI_API_KEY")
    set_env("LANGCHAIN_API_KEY")

    for s in super_graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Write a brief research report on the North American sturgeon. Include a chart."
                )
            ],
        },
        {"recursion_limit": 10},
    ):
        if "__end__" not in s:
            print(s)
            print("---")
