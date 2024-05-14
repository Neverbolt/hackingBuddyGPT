import abc
import inspect
from typing import Union, Type, Dict, Iterable

import openai
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.completion_create_params import Function
from pydantic import create_model, BaseModel


class Capability(abc.ABC):
    """
    A capability is something that can be used by an LLM to perform a task.
    The method signature for the __call__ method is not yet defined, but it will probably be different for different
    types of capabilities (though it is recommended to have the same signature for capabilities, that accomplish the
    same task but slightly different / for a different target).

    At the moment, this is not yet a very powerful class, but in the near-term future, this will provide an automated
    way of providing a json schema for the capabilities, which can then be used for function-calling LLMs.
    """
    @abc.abstractmethod
    def describe(self, name: str = None) -> str:
        """
        describe should return a string that describes the capability. This is used to generate the help text for the
        LLM.
        I don't like, that at the moment the name under which the capability is available to the LLM is allowed to be
        passed in, but it is necessary at the moment, to be backwards compatible. Please do not use the name if you
        don't really have to, then we can see if we can remove it in the future.

        This is a method and not just a simple property on purpose (though it could become a @property in the future, if
        we don't need the name parameter anymore), so that it can template in some of the capabilities parameters into
        the description.
        """
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        The actual execution of a capability, please make sure, that the parameters and return type of your
        implementation are well typed, as this will make it easier to support full function calling soon.
        """
        pass

    def to_model(self, name: str) -> BaseModel:
        """
        Converts the parameters of the `__call__` function of the capability to a pydantic model, that can be used to
        interface with an LLM using eg instructor or the openAI function calling API.
        The model will have the same name as the capability class and will have the same fields as the `__call__`,
        the `__call__` method can then be accessed by calling the `execute` method of the model.
        """
        sig = inspect.signature(self.__call__)
        fields = {param: (param_info.annotation, param_info.default if param_info.default is not inspect._empty else ...) for param, param_info in sig.parameters.items()}
        model_type = create_model(self.__class__.__name__, __doc__=self.describe(name), **fields)

        def execute(model):
            return self(**model.dict())
        model_type.execute = execute

        return model_type


# An Action is the base class to allow proper typing information of the generated class in `capabilities_to_action_mode`
# This description should not be moved into a docstring inside the class, as it will otherwise be provided in the LLM prompt
class Action(BaseModel):
    action: BaseModel

    def execute(self):
        return self.action.execute()


def capabilities_to_action_model(capabilities: Dict[str, Capability]) -> Type[Action]:
    """
    When one of multiple capabilities should be used, then an action model can be created with this function.
    This action model is a pydantic model, where all possible capabilities are represented by their respective models in
    a union type for the action field.
    This allows the LLM to define an action to be used, which can then simply be called using the `execute` function on
    the model returned from here.
    """
    class Model(Action):
        action: Union[tuple([capability.to_model(name) for name, capability in capabilities.items()])]

    return Model


def capabilities_to_functions(capabilities: Dict[str, Capability]) -> Iterable[openai.types.chat.completion_create_params.Function]:
    """
    This function takes a dictionary of capabilities and returns a dictionary of functions, that can be called with the
    parameters of the respective capabilities.
    """
    return [
        Function(name=name, description=capability.describe(name), parameters=capability.to_model(name).schema())
        for name, capability in capabilities.items()
    ]


def capabilities_to_tools(capabilities: Dict[str, Capability]) -> Iterable[openai.types.chat.completion_create_params.ChatCompletionToolParam]:
    """
    This function takes a dictionary of capabilities and returns a dictionary of functions, that can be called with the
    parameters of the respective capabilities.
    """
    return [
        ChatCompletionToolParam(type="function", function=Function(name=name, description=capability.describe(name), parameters=capability.to_model(name).schema()))
        for name, capability in capabilities.items()
    ]

