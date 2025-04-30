import asyncio

from agents import Agent, Runner, WebSearchTool
from agents.model_settings import ModelSettings

INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succinctly, no need to have complete sentences or good "
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary "
    "itself."
)

search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
)

agent = Agent(
    name="Search&Reason",
    model="o4-mini",  # o3 でも OK（速度か精度で選択）
    instructions=(
        "カスターサポートの相談員として、ユーザーの質問に回答してください。"
        "追加の情報が必要であれば顧客に聞き返したりWeb検索してください。"
        "電話なので短く返してください"
    ),
    # tools=[WebSearchTool()],  # Hosted tool。位置情報を渡すことも可
    tools=[
        search_agent.as_tool(
            tool_name="search_web",
            tool_description="Webで検索する。",
        ),
    ],
)


async def main():
    result = None
    while True:
        question = input("> ")
        result = await Runner.run(
            agent,
            question,
            previous_response_id=result.last_response_id if result else None,
        )
        print("---- 最終出力 ----")
        print(result.final_output)


asyncio.run(main())
