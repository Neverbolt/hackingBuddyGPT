from dataclasses import dataclass
from typing import override

from hackingBuddyGPT.capabilities import Capability
from hackingBuddyGPT.utils.limits import Limits


@dataclass
class UserInputCapability(Capability):
    limits: Limits

    @override
    def describe(self) -> str:
        return (
            "The user can not directly communicate with you, use this capability to get user input.\n"
            "This needs to be used whenever you're done with a task and want to get the next one!"
        )

    @override
    async def __call__(self, prompt: str) -> str:
        try:
            print(prompt)
            return input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            self.limits.complete()
            return "user aborted"
