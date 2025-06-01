"""Gymnax: A library for creating and registering Gym environments."""
from beartype.claw import beartype_this_package

beartype_this_package()

from gymnax import environments, registration


EnvParams = environments.EnvParams
EnvState = environments.EnvState
make = registration.make
registered_envs = registration.registered_envs


__all__ = ["make", "registered_envs", "EnvState", "EnvParams"]
