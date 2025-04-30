import asyncio

from agents import Agent, Runner

agent = Agent(
    name="assistant",
    model="gpt-4o-mini",  # o3 でも OK（速度か精度で選択）
    instructions=("あなたは親切なアシスタントです"),
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
