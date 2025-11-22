from typing import override

from hackingBuddyGPT.usecases.agents import Agent
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.utils.limits import Limits
from hackingBuddyGPT.utils.openai.openai_lib import OpenAILib


class TextCompletionAgent(Agent):
    llm: OpenAILib = None

    @override
    async def perform_round(self, limits: Limits):
        pass  # TODO: implement


@use_case("Text Completion")
class TextCompletionUseCase(AutonomousAgentUseCase[TextCompletionAgent]):
    pass
