import argparse
import dataclasses
import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict

from dotenv import load_dotenv

from typing import Type


load_dotenv()


def parameter(*, desc: str, default=dataclasses.MISSING, init: bool = True, repr: bool = True, hash=None,
              compare: bool = True, metadata: Dict = None, kw_only: bool = dataclasses.MISSING) -> dataclasses.Field:
    if metadata is None:
        metadata = dict()
    metadata["desc"] = desc

    return dataclasses.field(default=default, default_factory=dataclasses.MISSING, init=init, repr=repr, hash=hash,
                             compare=compare, metadata=metadata, kw_only=kw_only)


def get_default(key, default):
    return os.getenv(key, os.getenv(key.upper(), os.getenv(key.replace(".", "_"), os.getenv(key.replace(".", "_").upper(), default))))


@dataclass
class ParameterDefinition:
    """
    A ParameterDefinition is used for any parameter that is just a simple type, which can be handled by argparse directly.
    """
    name: str
    type: Type
    default: Any
    description: str

    def parser(self, basename: str, parser: argparse.ArgumentParser):
        name = f"{basename}{self.name}"
        default = get_default(name, self.default)

        parser.add_argument(f"--{name}", type=self.type, default=default, required=default is None,
                            help=self.description)

    def get(self, basename: str, args: argparse.Namespace):
        return getattr(args, f"{basename}{self.name}")


ParameterDefinitions = Dict[str, ParameterDefinition]


@dataclass
class ComplexParameterDefinition(ParameterDefinition):
    """
    A ComplexParameterDefinition is used for any parameter that is a complex type (which itself only takes simple types,
    or other types that fit the ComplexParameterDefinition), requiring a recursive build_parser.
    It is important to note, that at some point, the parameter must be a simple type, so that argparse (and we) can handle
    it. So if you have recursive type definitions that you try to make configurable, this will not work.
    """
    parameters: ParameterDefinitions

    def parser(self, basename: str, parser: argparse.ArgumentParser):
        for name, parameter in self.parameters.items():
            if isinstance(parameter, dict):
                build_parser(parameter, parser, f"{basename}{self.name}.")
            else:
                parameter.parser(f"{basename}{self.name}.", parser)

    def get(self, basename: str, args: argparse.Namespace):
        parameter = self.type(**get_arguments(self.parameters, args, f"{basename}{self.name}."))
        if hasattr(parameter, "init"):
            parameter.init()
        return parameter


def get_class_parameters(cls, name: str = None, fields: Dict[str, dataclasses.Field] = None) -> ParameterDefinitions:
    if name is None:
        name = cls.__name__
    if fields is None and hasattr(cls, "__dataclass_fields__"):
        fields = cls.__dataclass_fields__
    return get_parameters(cls.__init__, name, fields)


def get_parameters(fun, basename: str, fields: Dict[str, dataclasses.Field] = None) -> ParameterDefinitions:
    if fields is None:
        fields = dict()

    sig = inspect.signature(fun)
    params: ParameterDefinitions = {}
    for name, param in sig.parameters.items():
        if name == "self" or name.startswith("_"):
            continue

        if not param.annotation:
            raise ValueError(f"Parameter {name} of {basename}.{fun.__name__} must have a type annotation")

        default = param.default if param.default != inspect.Parameter.empty else None
        description = None
        type = param.annotation

        field = None
        if isinstance(default, dataclasses.Field):
            field = default
            default = field.default
        elif name in fields:
            field = fields[name]

        if field is not None:
            description = field.metadata.get("desc", None)
            if field.type is not None:
                type = field.type

        if hasattr(type, "__parameters__"):
            params[name] = ComplexParameterDefinition(name, type, default, description, get_class_parameters(type, f"{basename}.{fun.__name__}"))
        elif type in (str, int, bool):
            params[name] = ParameterDefinition(name, type, default, description)
        else:
            raise ValueError(f"Parameter {name} of {basename}.{fun.__name__} must have str, int, bool, or a __parameters__ class as type, not {type}")

    return params


def build_parser(parameters: ParameterDefinitions, parser: argparse.ArgumentParser, basename: str = ""):
    for name, parameter in parameters.items():
        parameter.parser(basename, parser)


def get_arguments(parameters: ParameterDefinitions, args: argparse.Namespace, basename: str = "") -> Dict[str, Any]:
    return {name: parameter.get(basename, args) for name, parameter in parameters.items()}


Configurable = Type  # TODO: Define type


def configurable(service_name: str, service_desc: str):
    """
    Anything that is decorated with the @configurable decorator gets the parameters of its __init__ method extracted,
    which can then be used with build_parser and get_arguments to recursively prepare the argparse parser and extract the
    initialization parameters. These can then be used to initialize the class with the correct parameters.
    """
    def inner(cls) -> Configurable:
        cls.name = service_name
        cls.description = service_desc
        cls.__service__ = True
        cls.__parameters__ = get_class_parameters(cls)

        return cls

    return inner
