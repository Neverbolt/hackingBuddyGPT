from typing import override

from hackingBuddyGPT.capabilities import SSHRunCommand
from hackingBuddyGPT.usecases.agents import ChatAgent
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.usecases.bsides.utils import UserInputCapability
from hackingBuddyGPT.utils import SSHConnection
from hackingBuddyGPT.utils.limits import Limits


class ShellAccessAgent(ChatAgent):
    @override
    async def system_message(self, limits: Limits) -> str:
        pass  # TODO: copy over

    @override
    async def before_run(self, limits: Limits):
        pass  # TODO: copy over and implement


# @use_case("Shell Access")
class ShellAccessUseCase(AutonomousAgentUseCase[ShellAccessAgent]):
    pass
