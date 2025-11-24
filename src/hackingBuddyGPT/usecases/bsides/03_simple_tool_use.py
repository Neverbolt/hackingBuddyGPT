from dataclasses import field
from typing import Literal, override

from hackingBuddyGPT.capabilities import Capability
from hackingBuddyGPT.capabilities.capability import OptimizedSchemaGenerator
from hackingBuddyGPT.usecases.agents import Agent, Prompt
from hackingBuddyGPT.usecases.base import AutonomousAgentUseCase, use_case
from hackingBuddyGPT.utils.limits import Limits
from hackingBuddyGPT.utils.openai.openai_lib import OpenAILib


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


class SimpleToolsAgent(Agent):
    llm: OpenAILib = None

    _prompt_history: Prompt = field(default_factory=list)

    @override
    async def before_run(self, limits: Limits):
        self.add_capability(CalculatorCapability())

    @override
    async def perform_round(self, limits: Limits):
        import pprint

        query = input("> ")
        self._prompt_history.append({"role": "user", "content": query})

        # pprint.pprint(self._prompt_history)
        result = self.llm.get_response(
            self._prompt_history,
            capabilities=self._capabilities,
        )
        print("<", result.result.content)
        self._prompt_history.append(result.result)

        tool_call_results = await self.run_tool_calls(0, result.result)
        for tool_call_result in tool_call_results:
            self._prompt_history.append(tool_call_result)
            print(tool_call_result)


@use_case("Simple Tools")
class SimpleToolsUseCase(AutonomousAgentUseCase[SimpleToolsAgent]):
    pass
