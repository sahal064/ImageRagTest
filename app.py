from agent.image_agent import create_image_agent
from tools.image_search_tool import image_search_tool
from tools.chart_tool import chart_reasoning_tool
from langchain_core.messages import ToolMessage,HumanMessage,AIMessage

agent=create_image_agent()
TOOLS = {
    "image_search_tool": image_search_tool,
    "chart_reasoning_tool": chart_reasoning_tool
}
def run_agent(query:str):
    messages = [HumanMessage(content=query)]


    while True:
       

        response=agent.invoke({
            "input":messages
        })


        if not response.tool_calls:
            print(" \n This is the Final Answer: \n",response.content)
            break

        for call in response.tool_calls:
            tool_name=call["name"]
            tool_args=call["args"]
            if tool_name == "image_search_tool":
                image_metadata = image_search_tool.invoke(tool_args)
                tool_output = image_metadata
                print(f"This is the Tool output for searching {tool_output}")

            elif tool_name == "chart_reasoning_tool":
                tool_args["image_metadata"] = image_metadata
                tool_output = chart_reasoning_tool.invoke(tool_args)
                print(f"This is the Tool_Output for reasoning{tool_output}")
          
            messages.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    content=str(tool_output)
                )
            )
            query=messages[-1].content

        messages.append(
            HumanMessage(
                content=f"Using the above tool results, answer all the above questions."
            )
        )



        print("\n Agent Output: \n", response.content)


if __name__ == "__main__":
    while True:
        q = input("\nUser: ")
        if q.lower() == "exit":
            break
        run_agent(q)