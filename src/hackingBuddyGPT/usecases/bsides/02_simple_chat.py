from dataclasses import field
from typing import override

from hackingBuddyGPT.usecases.agents import Agent, Prompt
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.utils.limits import Limits
from hackingBuddyGPT.utils.openai.openai_lib import OpenAILib


class SimpleChatAgent(Agent):
    llm: OpenAILib = None

    @override
    async def perform_round(self, limits: Limits):
        pass  # TODO: copy over and implement


@use_case("Simple Chat")
class SimpleChatUseCase(AutonomousAgentUseCase[SimpleChatAgent]):
    pass
