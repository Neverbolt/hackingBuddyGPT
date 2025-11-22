from dataclasses import field
from typing import override

from hackingBuddyGPT.usecases.agents import Agent, Prompt
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.utils.limits import Limits
from hackingBuddyGPT.utils.openai.openai_lib import OpenAILib


class SimpleChatAgent(Agent):
    llm: OpenAILib = None

    _prompt_history: Prompt = field(default_factory=list)

    @override
    async def perform_round(self, limits: Limits):
        try:
            query = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            limits.complete()
            return

        self._prompt_history.append({"role": "user", "content": query})

        # print(self._prompt_history)
        result = self.llm.get_response(self._prompt_history)
        self._prompt_history.append(result.result)
        print("<", result.result.content)

        limits.register_message(result)
        limits.register_round()


@use_case("Simple Chat")
class SimpleChatUseCase(AutonomousAgentUseCase[SimpleChatAgent]):
    pass
