from __future__ import annotations
from alrd.agent.absagent import Agent, AgentReset
from alrd.agent.gp import (
    RandomGPAgent,
    PiecewiseRandomGPAgent,
    create_async_rbf_gp_agent,
)
from alrd.agent.uniform import UniformAgent
from alrd.agent.keyboard import KeyboardAgent, KeyboardGPAgent, KeyboardResetAgent
from alrd.agent.xbox import SpotXbox2D
from alrd.agent.trajaxopt import TraJaxOptAgent
from alrd.agent.adapter import AgentAdapter
from alrd.agent.model_based import ModelBasedAgentAdapter
from alrd.agent.sac import SACAgent
from alrd.spot_gym.utils.utils import MAX_ANGULAR_SPEED, MAX_SPEED
from opax.agents.model_based.model_based_agent import ModelBasedAgent
from opax.optimizers.sac_based_optimizer import SACOptimizer
from enum import Enum
import pickle
import jax
import cloudpickle


class AgentType(Enum):
    KEYBOARD = "keyboard"
    KEYBOARD_GP = "keyboard_gp"
    KEYBOARD_GP_XY = "keyboard_gp_xy"
    SAC = "sac"
    ILQROPEN = "ilqropen"

    def __call__(self, args):
        if self == AgentType.KEYBOARD:
            return KeyboardAgent(
                xy_speed=args.xy_speed, a_speed=args.a_speed, noangle=args.noangle
            )
        elif self == AgentType.KEYBOARD_GP or self == AgentType.KEYBOARD_GP_XY:
            if self == AgentType.KEYBOARD_GP_XY:
                scale = (args.xy_speed, args.xy_speed, 0.0)
            else:
                scale = (args.xy_speed, args.xy_speed, 0.5 * args.a_speed / 120.0)
            gp_agent = create_async_rbf_gp_agent(
                length_scale=args.length_scale,
                noise=args.noise,
                scale=scale if not args.noangle else scale[:2],
                max_steps=(
                    args.episode_len if args.episode_len is not None else args.freq * 60
                ),
                freq=args.freq
                // (args.repeat_action if args.repeat_action is not None else 1),
                sample=args.gp_undersample,
                seed=args.seed,
            )
            return KeyboardGPAgent(
                gp_agent=gp_agent, xy_speed=args.xy_speed, a_speed=args.a_speed
            )
        elif self == AgentType.SAC:
            with open(args.agent_checkpoint, "rb") as f:
                checkpoint = pickle.load(f)
            agent, opt_state = checkpoint.create_agent()
            return AgentAdapter(agent, rng=args.agent_rng)
        elif self == AgentType.ILQROPEN:
            with open(args.model_checkpoint, "rb") as f:
                checkpoint = pickle.load(f)
            checkpoint.args["reward_model"] = args.reward_model
            model = checkpoint.create_model()
            print("args rng", args.agent_rng)
            return TraJaxOptAgent.create(
                model=model, action_dim=(3,), horizon=args.horizon, rng=args.agent_rng
            )
        else:
            raise NotImplementedError(f"Agent type {self} not implemented")


class SpotAgentEnum(Enum):
    KEYBOARD = "keyboard"
    XBOX = "xbox"
    SAC = "sac"
    SACMB = "sacmb"
    GP = "gp"
    UNIFORM = "uniform"


def create_spot_agent(
    observation_space,
    action_space,
    agent_type: SpotAgentEnum,
    optimizer_path: str | None,
    smoothing_coeff: float | None,
    rng,
    explore: bool,
    episode_len: int | None,
    freq: float,
    gp_undersample: int | None,
):
    if agent_type == SpotAgentEnum.KEYBOARD:
        base_agent = KeyboardAgent(xy_speed=1, a_speed=1)
        return KeyboardResetAgent(base_agent)
    elif agent_type == SpotAgentEnum.XBOX:
        return SpotXbox2D()
    elif agent_type == SpotAgentEnum.SAC:
        sac_optimizer = cloudpickle.load(open(optimizer_path, "rb"))
        assert isinstance(sac_optimizer, SACOptimizer)
        agent = SACAgent(sac_optimizer, smoothing_coeff)
        return agent
    elif agent_type == SpotAgentEnum.SACMB:
        sac_optimizer = cloudpickle.load(open(optimizer_path, "rb"))
        assert isinstance(sac_optimizer, SACOptimizer)
        sac_kwargs = {
            "actor_features": sac_optimizer.agent_list[0].actor.features,
            "critic_features": sac_optimizer.agent_list[0].critic.features,
        }
        agent = ModelBasedAgent(
            action_space=action_space,
            observation_space=observation_space,
            dynamics_model=sac_optimizer.dynamics_model,
            policy_optimizer_name="SacOpt",
            optimizer_kwargs={"sac_kwargs": sac_kwargs},
        )
        agent.update_optimizer(sac_optimizer)
        return ModelBasedAgentAdapter(agent, rng, eval=not explore)
    elif agent_type == SpotAgentEnum.GP:
        scale = (MAX_SPEED / 2.0, MAX_SPEED / 2.0, MAX_ANGULAR_SPEED / 2.0)
        gp_agent = create_async_rbf_gp_agent(
            length_scale=1.0,
            noise=1e-2,
            scale=scale,
            max_steps=episode_len,
            freq=freq,
            sample=gp_undersample,
            seed=jax.random.randint(
                rng,
                [
                    1,
                ],
                minval=0,
                maxval=2**31 - 1,
            ).item(),
        )
        return gp_agent
    elif agent_type == SpotAgentEnum.UNIFORM:
        return UniformAgent(
            jax.random.randint(
                rng,
                [
                    1,
                ],
                minval=0,
                maxval=2**31 - 1,
            ).item()
        )
    else:
        raise NotImplementedError(f"Agent type {agent_type} not implemented")


__all__ = [
    "Agent",
    "AgentReset",
    "RandomGPAgent",
    "PiecewiseRandomGPAgent",
    "KeyboardAgent",
]
