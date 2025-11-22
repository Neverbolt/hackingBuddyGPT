from typing import override

from hackingBuddyGPT.capabilities import SSHRunCommand
from hackingBuddyGPT.capabilities.capability import capability_list_to_dict
from hackingBuddyGPT.usecases.agents import ChatAgent, SubAgentCapability
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.usecases.bsides.utils import UserInputCapability
from hackingBuddyGPT.utils import SSHConnection
from hackingBuddyGPT.utils.limits import Limits


class MultiAgentAgent(ChatAgent):
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
            SubAgentCapability(
                ChatAgent,
                self.llm,
                self.log,
                limits,
                capability_list_to_dict([SSHRunCommand(conn=self.conn)]),
                "subagent",
            )
        )

        self._prompt_history.append({"role": "user", "content": input("Initial message: ")})


@use_case("Multi Agent")
class MultiAgentUseCase(AutonomousAgentUseCase[MultiAgentAgent]):
    pass
