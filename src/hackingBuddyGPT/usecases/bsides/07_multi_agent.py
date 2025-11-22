from typing import override

from hackingBuddyGPT.capabilities import SSHRunCommand
from hackingBuddyGPT.capabilities.capability import capability_list_to_dict
from hackingBuddyGPT.usecases.agents import ChatAgent, SubAgentCapability
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.usecases.bsides.utils import UserInputCapability
from hackingBuddyGPT.utils import SSHConnection
from hackingBuddyGPT.utils.limits import Limits


class MultiAgentAgent(ChatAgent):
    @override
    async def system_message(self, limits: Limits) -> str:
        pass  # TODO: copy over

    @override
    async def before_run(self, limits: Limits):
        pass  # TODO: copy over and implement


@use_case("Multi Agent")
class MultiAgentUseCase(AutonomousAgentUseCase[MultiAgentAgent]):
    pass
