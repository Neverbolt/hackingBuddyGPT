from typing import override

from hackingBuddyGPT.usecases.agents import Agent
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.utils.limits import Limits
from hackingBuddyGPT.utils.openai.openai_lib import OpenAILib


class TextCompletionAgent(Agent):
    llm: OpenAILib = None

    @override
    async def perform_round(self, limits: Limits):
        try:
            query = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            limits.complete()
            return

        result = self.llm.get_response(query)
        # print(result)
        print("<", result.result.content)

        limits.register_message(result)
        limits.register_round()


@use_case("Text Completion")
class TextCompletionUseCase(AutonomousAgentUseCase[TextCompletionAgent]):
    pass
