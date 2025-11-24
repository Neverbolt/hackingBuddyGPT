from dataclasses import field
from typing import override

from pydantic.types import SecretType

from hackingBuddyGPT.usecases.agents import Agent, Prompt
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.utils.limits import Limits
from hackingBuddyGPT.utils.openai.openai_lib import OpenAILib


class SimpleChatAgent(Agent):
    llm: OpenAILib = None

    _prompt_history: Prompt = field(default_factory=list)

    @override
    async def perform_round(self, limits: Limits):
        import pprint

        query = input("> ")
        self._prompt_history.append({"role": "user", "content": query})

        pprint.pprint(self._prompt_history)
        result = self.llm.get_response(self._prompt_history)
        self._prompt_history.append(result.result)

        print("<", result.result.content)
        # pprint.pprint(result)


@use_case("Simple Chat")
class SimpleChatUseCase(AutonomousAgentUseCase[SimpleChatAgent]):
    pass
