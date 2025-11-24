from typing import override

from hackingBuddyGPT.capabilities.http_request import HTTPRequest
from hackingBuddyGPT.usecases.agents import ChatAgent
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.usecases.bsides.utils import UserInputCapability
from hackingBuddyGPT.utils import parameter
from hackingBuddyGPT.utils.limits import Limits


class WebExplorationAgent(ChatAgent):
    base_url: str = parameter(
        desc="Base URL that the web requests are bound to (should include protocol and domain, no trailing slash)",
        default=None,
    )

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
        self.add_capability(HTTPRequest(self.base_url))

        self._prompt_history.append({"role": "user", "content": input("Initial message: ")})


@use_case("Web Exploration")
class WebExplorationUseCase(AutonomousAgentUseCase[WebExplorationAgent]):
    pass
