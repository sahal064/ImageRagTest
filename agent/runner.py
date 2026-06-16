from agent.image_agent import create_image_agent
from tools.image_search_tool import image_search_tool
from tools.chart_tool import chart_reasoning_tool
from langchain_core.messages import ToolMessage, HumanMessage

agent = create_image_agent()

def run_agent(query: str, top_k: int = 5):
    messages = [HumanMessage(content=query)]
    image_metadata = None
    final_answer = None
    used_images = []

    while True:
        response = agent.invoke({"input": messages})

        # ✅ Final answer
        if not response.tool_calls:
            final_answer = response.content
            break

        for call in response.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]

            if tool_name == "image_search_tool":
                tool_args["top_k"] = top_k
                results = image_search_tool.invoke(tool_args)
                image_metadata = results[0] if results else {}
                used_images.extend(results)
                tool_output = results

            elif tool_name == "chart_reasoning_tool":
                tool_args["image_metadata"] = image_metadata
                tool_output = chart_reasoning_tool.invoke(tool_args)

            messages.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    content=str(tool_output)
                )
            )

        messages.append(
            HumanMessage(
                content="Using the above tool results, answer the original question clearly."
            )
        )

    return {
        "answer": final_answer,
        "images": used_images
    }
