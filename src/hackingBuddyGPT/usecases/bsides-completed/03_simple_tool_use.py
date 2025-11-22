from dataclasses import field
from typing import Literal, override

from hackingBuddyGPT.capabilities import Capability
from hackingBuddyGPT.usecases.agents import Agent, Prompt
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.utils.limits import Limits
from hackingBuddyGPT.utils.openai.openai_lib import OpenAILib


class CalculatorCapability(Capability):
    @override
    def describe(self) -> str:
        return "LLMs are bad at calculating, use this calculator instead. Supported operations are: addition (+), subtraction (-), multiplication (*) and division (/)."

    @override
    async def __call__(self, a: int, operation: Literal["+", "-", "*", "/"], b: int) -> str:
        match operation:
            case "+":
                return str(a + b)
            case "-":
                return str(a - b)
            case "*":
                return str(a * b)
            case "/":
                return str(a / b)
            case _:
                return "Invalid operation"


class SimpleToolsAgent(Agent):
    llm: OpenAILib = None

    _prompt_history: Prompt = field(default_factory=list)
    _capabilities: dict[str, Capability] = field(default_factory=dict)

    @override
    async def before_run(self, limits: Limits):
        self.add_capability(CalculatorCapability())

    @override
    async def perform_round(self, limits: Limits):
        try:
            query = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            limits.complete()
            return

        self._prompt_history.append({"role": "user", "content": query})

        # print(self._prompt_history)
        result = self.llm.get_response(self._prompt_history, capabilities=self._capabilities)
        self._prompt_history.append(result.result)
        # print(result)
        print("<", result.result.content)

        tool_call_results = await self.run_tool_calls(0, result.result)
        for tool_call_result in tool_call_results:
            self._prompt_history.append(tool_call_result)
            # print(tool_call_result)

        limits.register_message(result)
        limits.register_round()
        # 158745-1854*152


@use_case("Simple Tools")
class SimpleToolsUseCase(AutonomousAgentUseCase[SimpleToolsAgent]):
    pass
