from dataclasses import dataclass
from typing import Literal, override

from hackingBuddyGPT.capabilities import Capability
from hackingBuddyGPT.usecases.agents import ChatAgent
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.utils.limits import Limits


class CalculatorCapability(Capability):
    def describe(self):
        return "A calculator that can perform basic arithmetic operations."

    async def __call__(self, a: int, operator: Literal["+", "-", "*", "/"], b: int) -> str:
        if operator == "+":
            return str(a + b)
        elif operator == "-":
            return str(a - b)
        elif operator == "*":
            return str(a * b)
        elif operator == "/":
            return str(a / b)
        else:
            raise ValueError(f"Invalid operator: {operator}")


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


class SimplifiedChatAgent(ChatAgent):
    @override
    async def system_message(self, limits: Limits) -> str:
        return "You are a helpful assistant that is always looking out for the user."

    @override
    async def add_limits_message(self, limits: Limits):
        pass  # do not need this yet and it just buries the output

    @override
    async def before_run(self, limits: Limits):
        await super().before_run(limits)

        self.add_capability(CalculatorCapability())
        self.add_capability(UserInputCapability(limits))

        self._prompt_history.append({"role": "user", "content": input("Initial message: ")})


@use_case("Simplified Chat")
class SimplifiedChatUseCase(AutonomousAgentUseCase[SimplifiedChatAgent]):
    pass
