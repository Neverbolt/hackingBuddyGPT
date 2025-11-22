from dataclasses import dataclass
from typing import Literal, override

from hackingBuddyGPT.capabilities import Capability
from hackingBuddyGPT.usecases.agents import ChatAgent
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.utils.limits import Limits


class CalculatorCapability(Capability):
    pass  # TODO: copy over


@dataclass
class UserInputCapability(Capability):
    limits: Limits

    @override
    def describe(self) -> str:
        pass  # TODO: implement

    @override
    async def __call__(self, prompt: str) -> str:
        pass  # TODO: implement


class SimplifiedChatAgent(ChatAgent):
    @override
    async def system_message(self, limits: Limits) -> str:
        pass  # TODO: copy over

    @override
    async def add_limits_message(self, limits: Limits):
        pass  # do not need this yet and it just buries the output

    @override
    async def before_run(self, limits: Limits):
        pass  # TODO: copy over and implement


# @use_case("Simplified Chat")
class SimplifiedChatUseCase(AutonomousAgentUseCase[SimplifiedChatAgent]):
    pass
