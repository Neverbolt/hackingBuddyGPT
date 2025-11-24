from typing import override

from hackingBuddyGPT.capabilities import SSHRunCommand
from hackingBuddyGPT.usecases.agents import ChatAgent
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.usecases.bsides.utils import UserInputCapability
from hackingBuddyGPT.utils import SSHConnection
from hackingBuddyGPT.utils.limits import Limits


class ShellAccessAgent(ChatAgent):
    conn: SSHConnection = None

    @override
    async def system_message(self, limits: Limits) -> str:
        return (
            "You are a helpful assistant. "
            "The user can not directly communicate with you other than the first message, use the UserInput capability to get user input.\n"
        )

    @override
    async def before_run(self, limits: Limits):
        await super().before_run(limits)
        self.add_capability(UserInputCapability(limits))
        self.add_capability(
            SSHRunCommand(
                conn=self.conn,
                additional_description="You can use this capability to run commands on a kali linux machine that is in the same network as the server you want to attack.",
            )
        )

        self._prompt_history.append({"role": "user", "content": input("Initial message: ")})


@use_case("Shell Access")
class ShellAccessUseCase(AutonomousAgentUseCase[ShellAccessAgent]):
    pass
