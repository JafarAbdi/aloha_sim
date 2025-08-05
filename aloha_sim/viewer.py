# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Interactive viewer for running model inference.

Instructions:

- I = enter new instruction.
- space bar = pause/restart.
- backspace = reset environment.
- mouse right moves the camera
- mouse left rotates the camera
- double click to select an object

When the environment is not running:
- ctrl + mouse left rotates a selected object
- ctrl + mouse right moves a selected object

When the environment is running:
- ctrl + mouse left applies torque to an object
- ctrl + mouse right applies force to an object
"""

import time
from typing import Any, Sequence, TypeAlias
import torch

import pathlib
from absl import app
from absl import flags
from absl import logging
from aloha_sim import task_suite
from dm_control import composer
import dm_env
from dm_env import specs
import mujoco
import mujoco.viewer
import numpy as np
from rich import console
from rich import panel
from rich import prompt
from rich import text
import tree
from aloha_sim.scripted_policies.marker import Policy as ScriptedPolicy
from robot_learning.constants import (
    CONFIG_NAME,
    DATASET_METADATA_FILE,
    DATASET_STATISTICS_FILE,
)
from robot_learning.data_normalizer import DataNormalizer
from robot_learning.dataset import DatasetMetadata, DatasetStatistics
from robot_learning.utils import pretty_print_omegaconfig
from omegaconf import OmegaConf, DictConfig
from aloha_sim.tasks.base.aloha2_task import HOME_CTRL_CLOSE, HOME_CTRL_OPEN
from torchvision.transforms import v2 as torchvision_transforms
import hydra
from robot_learning.model_utils import timestep_observation_to_tensor
from aloha_sim.trajectory_parameterization import (
    ToppraParameterizer,
    RuckigParameterizer,
    TotgParameterizer,
)
from aloha_sim.aloha import RESAMPLE_DT
from robot_learning.configs.train import TrainConfig


ActionSpec: TypeAlias = specs.Array
ExtraOutputSpec: TypeAlias = tree.Structure[specs.Array]
StateSpec: TypeAlias = tree.Structure[specs.Array]
State: TypeAlias = tree.Structure[np.typing.ArrayLike]
Action: TypeAlias = np.typing.ArrayLike
ExtraOutput: TypeAlias = tree.Structure[np.typing.ArrayLike]

_TASK_NAME = flags.DEFINE_enum(
    "task_name",
    "HandOverBanana",
    task_suite.TASK_FACTORIES.keys(),
    "Task name.",
)
_POLICY = flags.DEFINE_enum(
    "policy",
    "gemini_robotics_on_device",
    ["gemini_robotics_on_device", "no_policy", "scripted_policy", "trained_policy"],
    "Policy to use.",
)
_CHECKPOINT = flags.DEFINE_string(
    "checkpoint",
    "",
    "Path to the checkpoint file. If empty, the policy will not be restored.",
)

# --- Global State for Viewer Interaction ---
_GLOBAL_STATE = {
    "_IS_RUNNING": True,
    "_SHOULD_RESET": False,
    "_SINGLE_STEP": False,
    "_ASKING_INSTRUCTION": False,
}
_LOG_STEPS = 100
_DT = 0.02
_IMAGE_SIZE = (480, 848)
_ALOHA_CAMERAS = {
    "overhead_cam": _IMAGE_SIZE,
    "worms_eye_cam": _IMAGE_SIZE,
    "wrist_cam_left": _IMAGE_SIZE,
    "wrist_cam_right": _IMAGE_SIZE,
}
_ALOHA_JOINTS = {"joints_pos": 14}
_INIT_ACTION = np.asarray(
    [
        0.0,
        -0.96,
        1.16,
        0.0,
        -0.3,
        0.0,
        1.5,
        0.0,
        -0.96,
        1.16,
        0.0,
        -0.3,
        0.0,
        1.5,
    ]
)
_SERVE_ID = "gemini_robotics_on_device"


class NoPolicy:
    """A no-op policy that always returns the initial action."""

    def step(self, unused_observation: Any) -> np.ndarray:
        return _INIT_ACTION

    def reset(self) -> None:
        pass

    def set_task_instruction(self, unused_instruction: str) -> None:
        pass

    def setup(self) -> None:
        pass


class TrainedPolicy:
    """A no-op policy that always returns the initial action."""

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = pathlib.Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint directory {self.checkpoint} does not exist."
            )
        if not self.checkpoint.is_dir():
            raise ValueError(f"Checkpoint path {self.checkpoint} is not a directory.")
        self.current_index = 0
        self.current_actions = None

    @torch.no_grad()
    def step(self, observation: Any) -> np.ndarray:
        if self.current_actions is not None and self.current_index < len(
            self.current_actions
        ):
            action = self.current_actions[self.current_index]
            self.current_index += 1
            return np.concatenate(
                [HOME_CTRL_OPEN, action]
            )  # Return the first action in the batch
        batched_obs = timestep_observation_to_tensor(
            observation, self.dataset_metadata, "cuda"
        )

        for key, transform in self.proprio_preprocessing.items():
            if key in batched_obs:
                batched_obs[key] = transform(batched_obs[key])

        batched_obs = self.data_normalizer.normalize(
            batched_obs,
            [self.dataset_metadata.state_observation],
        )

        action_pred = self.model(batched_obs)

        denormalized_actions = self.data_normalizer.denormalize(
            action_pred,
            [self.dataset_metadata.action],
        )
        actions = (
            denormalized_actions[self.dataset_metadata.action].cpu().numpy().squeeze(0)
        )
        self.current_actions = actions
        max_velocity = np.ones_like(actions[0]) * np.pi
        max_acceleration = np.ones_like(actions[0]) * np.pi
        max_jerk = np.ones_like(actions[0])
        totg = TotgParameterizer(
            RESAMPLE_DT,
            max_velocity,
            max_acceleration,
        )
        toppra = ToppraParameterizer(
            RESAMPLE_DT,
            max_velocity,
            max_acceleration,
        )
        ruckig = RuckigParameterizer(
            RESAMPLE_DT,
            max_velocity,
            max_acceleration,
            max_jerk,
        )
        self.current_actions = totg.run(actions)
        self.current_index = 1
        return np.concatenate(
            [HOME_CTRL_OPEN, self.current_actions[0]]
        )  # Return the first action in the batch

    def reset(self) -> None:
        self.current_index = 0
        self.current_actions = None

    def set_task_instruction(self, unused_instruction: str) -> None:
        pass

    def setup(self) -> None:
        """Load the model from the checkpoint."""
        with hydra.initialize_config_dir(
            config_dir=str(pathlib.Path.cwd() / self.checkpoint), version_base=None
        ):
            scheme = OmegaConf.structured(TrainConfig)
            cfg = OmegaConf.merge(scheme, OmegaConf.load(self.checkpoint / CONFIG_NAME))
        pretty_print_omegaconfig(cfg)
        model_cls = hydra.utils.get_class(cfg.model._target_)
        model_params = {k: v for k, v in cfg.model.items() if k != "_target_"}
        # Instantiate model parameters if they are Hydra objects
        for k, v in model_params.items():
            if isinstance(v, dict | DictConfig) and "_target_" in v:
                model_params[k] = hydra.utils.instantiate(v)
        self.dataset_metadata = DatasetMetadata.from_file(
            self.checkpoint / DATASET_METADATA_FILE
        )
        dataset_statistics = DatasetStatistics.from_file(
            self.checkpoint / DATASET_STATISTICS_FILE,
        )
        self.data_normalizer = DataNormalizer(
            dataset_statistics,
            hydra.utils.instantiate(cfg.data_normalizer),
            device="cuda",
        )
        self.model = model_cls.from_pretrained(
            self.checkpoint,
            **model_params,
            dataset_meta=self.dataset_metadata,
        )
        self.image_preprocessing = (
            torchvision_transforms.Compose(
                hydra.utils.instantiate(cfg.image_preprocessing)
            )
            if cfg.image_preprocessing
            else None
        )
        self.proprio_preprocessing = (
            hydra.utils.instantiate(cfg.proprio_preprocessing)
            if cfg.proprio_preprocessing
            else None
        )


def _key_callback(key: int) -> None:
    """Viewer callbacks for key-presses."""
    if key == 32:  # Space bar
        _GLOBAL_STATE["_IS_RUNNING"] = not _GLOBAL_STATE["_IS_RUNNING"]
        logging.info("RUNNING = %s", _GLOBAL_STATE["_IS_RUNNING"])
    elif key == 259:  # Backspace
        _GLOBAL_STATE["_SHOULD_RESET"] = True
        logging.info("RESET = %s", _GLOBAL_STATE["_SHOULD_RESET"])
    elif key == 262:  # Right arrow
        _GLOBAL_STATE["_SINGLE_STEP"] = True
        _GLOBAL_STATE["_IS_RUNNING"] = True  # Allow one step to proceed
        logging.info("_SINGLE_STEP = %s", _GLOBAL_STATE["_SINGLE_STEP"])
    elif key == 73:  # I key
        _GLOBAL_STATE["_IS_RUNNING"] = False
        _GLOBAL_STATE["_ASKING_INSTRUCTION"] = True
    else:
        logging.info("UNKNOWN KEY PRESS = %s", key)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 2:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Initializing %s environment...", _TASK_NAME.value)
    if _TASK_NAME.value not in task_suite.TASK_FACTORIES.keys():
        raise ValueError(
            f"Unknown task_name: {_TASK_NAME.value}. Available tasks:"
            f" {list(task_suite.TASK_FACTORIES.keys())}"
        )
    task_class, kwargs = task_suite.TASK_FACTORIES[_TASK_NAME.value]
    task = task_class(
        cameras=_ALOHA_CAMERAS, control_timestep=_DT, update_interval=25, **kwargs
    )
    env = composer.Environment(
        task=task,
        time_limit=float("inf"),  # No explicit time limit from the environment
        random_state=np.random.RandomState(0),  # For reproducibility
        recompile_mjcf_every_episode=False,
        strip_singleton_obs_buffer_dim=True,
        delayed_observation_padding=composer.ObservationPadding.INITIAL_VALUE,
    )
    env.reset()

    # Instantiate the policy.
    if _POLICY.value == "no_policy":
        policy = NoPolicy()
    elif _POLICY.value == "scripted_policy":
        policy = ScriptedPolicy(env)
    elif _POLICY.value == "trained_policy":
        if not _CHECKPOINT.value:
            raise ValueError("Checkpoint path must be provided for trained_policy.")
        policy = TrainedPolicy(_CHECKPOINT.value)
        policy.setup()  # Load the model from the checkpoint
    else:
        try:
            print("Creating policy...")
            # Import safari_sdk only when needed for inference
            from safari_sdk.model import gemini_robotics_policy
            from safari_sdk.model import genai_robotics

            policy = gemini_robotics_policy.GeminiRoboticsPolicy(
                serve_id=_SERVE_ID,
                task_instruction=env.task.get_instruction(),
                inference_mode=gemini_robotics_policy.InferenceMode.SYNCHRONOUS,
                cameras=_ALOHA_CAMERAS,
                joints=_ALOHA_JOINTS,
                min_replan_interval=25,
                robotics_api_connection=genai_robotics.RoboticsApiConnectionType.LOCAL,
            )
            policy.setup()  # Initialize the policy
            print("GeminiRoboticsPolicy initialized successfully.")
        except ModuleNotFoundError:
            rich_console = console.Console()

            error_text = text.Text()
            error_text.append("🚨 ", style="bold red")
            error_text.append("safari_sdk not available!", style="bold red")

            help_text = text.Text()
            help_text.append(
                "To use inference features, install the inference dependencies:\n",
                style="yellow",
            )
            help_text.append(
                "   pip install aloha_sim[inference]\n\n", style="bold cyan"
            )
            help_text.append("Or run without inference using:\n", style="yellow")
            help_text.append(
                "   python aloha_sim/viewer.py --policy=no_policy", style="bold cyan"
            )

            error_panel = panel.Panel(
                help_text, title=error_text, border_style="red", padding=(1, 2)
            )

            rich_console.print(error_panel)
            raise
        except ValueError as e:
            print(f"Error initializing policy: {e}")
            raise
        except Exception as e:  # pylint: disable=broad-except
            print(f"An unexpected error occurred during initialization: {e}")
            raise

    logging.info("Running policy...")

    logging.info("Launching viewer...")
    viewer_model = env.physics.model.ptr
    viewer_data = env.physics.data.ptr
    with mujoco.viewer.launch_passive(
        viewer_model,
        viewer_data,
        key_callback=_key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer_handle:
        viewer_handle.sync()
        logging.info("Viewer started. Press Space to play/pause, Backspace to reset.")
        while viewer_handle.is_running():
            timestep = env.reset()
            policy.reset()
            instruction = task.get_instruction()
            viewer_handle.sync()

            steps = 0
            time_inference = 0
            time_stepping = 0
            sync_time = 0

            def is_done(timestep: dm_env.TimeStep) -> bool:
                """Check if the environment is done."""
                if _POLICY.value == "scripted_policy":
                    return policy.is_done()
                return timestep.last()

            while not is_done(timestep):
                # For debugging purposes, you can visualize the current frame
                # > viewer_handle.user_scn.ngeom = 0
                # > add_frame_to_scene(viewer_handle.user_scn, get_pick_pose(viewer_data))
                steps += 1
                if _GLOBAL_STATE["_ASKING_INSTRUCTION"]:
                    instruction = prompt.Prompt.ask(
                        "Enter new instruction. Press enter to use current instruction",
                        default=instruction,
                    )
                    policy.set_task_instruction(instruction)
                    logging.info("Using instruction: %s", instruction)
                    _GLOBAL_STATE["_ASKING_INSTRUCTION"] = False
                    _GLOBAL_STATE["_IS_RUNNING"] = True
                if _GLOBAL_STATE["_IS_RUNNING"] or _GLOBAL_STATE["_SINGLE_STEP"]:
                    frame_start_time = time.time()
                    try:
                        action = policy.step(timestep.observation)
                    except Exception as e:
                        logging.error("Policy step failed: %s", e)
                        input("Press Enter to continue...")
                    query_end_time = time.time()
                    time_inference += query_end_time - frame_start_time

                    current_timestep = env.step(action)
                    step_end_time = time.time()
                    time_stepping += step_end_time - query_end_time

                    viewer_handle.sync()

                    sync_time += time.time() - step_end_time

                    if steps % _LOG_STEPS == 0:
                        logging.info("Step: %s", steps)
                        logging.info(
                            "Inference time per step:\t%ss, total:\t%ss",
                            time_inference / _LOG_STEPS,
                            time_inference,
                        )
                        logging.info(
                            "Stepping time per step:\t%ss, total:\t%ss",
                            time_stepping / _LOG_STEPS,
                            time_stepping,
                        )
                        logging.info(
                            "Sync time per step:\t%ss, total:\t%ss",
                            sync_time / _LOG_STEPS,
                            sync_time,
                        )
                        time_inference = 0
                        time_stepping = 0
                        sync_time = 0

                    if _GLOBAL_STATE["_SHOULD_RESET"]:
                        # Reset was pressed mid-episode
                        _GLOBAL_STATE["_SHOULD_RESET"] = False
                        current_timestep = current_timestep._replace(
                            step_type=dm_env.StepType.LAST
                        )

                    assert (
                        not current_timestep.first()
                    ), "Environment auto-reseted mid-episode unexpectedly."
                    timestep = current_timestep

                    if _GLOBAL_STATE["_SINGLE_STEP"]:
                        _GLOBAL_STATE["_SINGLE_STEP"] = False
                        _GLOBAL_STATE["_IS_RUNNING"] = False  # Pause after single step

                with viewer_handle.lock():
                    # Apply perturbations if active (e.g. mouse drag)
                    if viewer_handle.perturb.active:
                        if _GLOBAL_STATE["_IS_RUNNING"]:
                            mujoco.mjv_applyPerturbForce(
                                env.physics.model.ptr,
                                env.physics.data.ptr,
                                viewer_handle.perturb,
                            )
                        else:
                            mujoco.mjv_applyPerturbPose(
                                env.physics.model.ptr,
                                env.physics.data.ptr,
                                viewer_handle.perturb,
                                flg_paused=1,
                            )
                            mujoco.mj_kinematics(
                                env.physics.model.ptr, env.physics.data.ptr
                            )
                    viewer_handle.sync()

                if not _GLOBAL_STATE["_IS_RUNNING"]:
                    time.sleep(0.01)  # Yield to other threads if paused

            if _GLOBAL_STATE[
                "_SHOULD_RESET"
            ]:  # Reset pressed at the very end of an episode
                _GLOBAL_STATE["_SHOULD_RESET"] = False
    logging.info("Viewer exited.")
    env.close()


if __name__ == "__main__":
    app.run(main)
