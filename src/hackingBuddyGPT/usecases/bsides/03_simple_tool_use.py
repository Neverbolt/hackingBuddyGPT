from dataclasses import field
from typing import Literal, override

from hackingBuddyGPT.capabilities import Capability
from hackingBuddyGPT.usecases.agents import Agent, Prompt
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.utils.limits import Limits
from hackingBuddyGPT.utils.openai.openai_lib import OpenAILib


class CalculatorCapability(Capability):
    pass  # TODO: implement


class SimpleToolsAgent(Agent):
    llm: OpenAILib = None

    _prompt_history: Prompt = field(default_factory=list)

    @override
    async def before_run(self, limits: Limits):
        pass  # TODO: implement

    @override
    async def perform_round(self, limits: Limits):
        pass  # TODO: copy over and implement


@use_case("Simple Tools")
class SimpleToolsUseCase(AutonomousAgentUseCase[SimpleToolsAgent]):
    pass
