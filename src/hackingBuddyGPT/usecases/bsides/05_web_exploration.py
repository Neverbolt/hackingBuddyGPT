from typing import override

from hackingBuddyGPT.capabilities.http_request import HTTPRequest
from hackingBuddyGPT.usecases.agents import ChatAgent
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.usecases.bsides.utils import UserInputCapability
from hackingBuddyGPT.utils import parameter
from hackingBuddyGPT.utils.limits import Limits


class WebExplorationAgent(ChatAgent):
    @override
    async def system_message(self, limits: Limits) -> str:
        pass  # TODO: copy over

    @override
    async def before_run(self, limits: Limits):
        pass  # TODO: copy over and implement


# @use_case("Web Exploration")
class WebExplorationUseCase(AutonomousAgentUseCase[WebExplorationAgent]):
    pass
