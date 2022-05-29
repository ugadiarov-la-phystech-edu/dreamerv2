from pathlib import Path

import cv2
import gym
import numpy as np
import yaml
from griddly import GymWrapper
from griddly.util.action_space import MultiAgentActionSpace

from dreamerv2.grafter.grafter.level_generators.crafter_generator import CrafterLevelGenerator


class GrafterWrapper(gym.Wrapper):
    def __init__(
            self,
            width,
            height,
            player_count=1,
            generator_seed=100,
            player_observer_type="PlayerSprite2D",
            global_observer_type="GlobalSprite2D",
            level_id=None
    ):

        current_file = Path(__file__).parent
        with open(current_file.joinpath("gdy").joinpath("grafter_base.yaml")) as f:
            gdy = yaml.load(f)
            gdy["Environment"]["Player"]["Count"] = player_count

            yaml_string = yaml.dump(gdy)

        self._level_id = level_id

        self._genv = GymWrapper(
            yaml_string=yaml_string,
            global_observer_type=global_observer_type,
            player_observer_type=player_observer_type,
            gdy_path=str(current_file.joinpath("gdy")),
            image_path=str(current_file.joinpath("assets")),
            level=level_id
        )

        if self._level_id is None:
            self._generator = CrafterLevelGenerator(
                generator_seed, width, height, self._genv.player_count
            )

        self._genv.reset()

        super().__init__(self._genv)

        # flatten the action space
        self.action_space, self.flat_action_mapping = self._flatten_action_space()
        self.resize_shape = (96, 96)
        if self.resize_shape:
            self.observation_space = gym.spaces.Box(0, 255, (*self.resize_shape, 3), dtype=np.uint8)

    def _process_observation(self, observation):
        observation = np.swapaxes(observation, 0, 2)
        if self.resize_shape:
            observation = cv2.resize(observation, dsize=self.resize_shape, interpolation=cv2.INTER_AREA)

        return observation

    def _flatten_action_space(self):
        flat_action_mapping = []
        actions = []
        actions.append("NOP")
        flat_action_mapping.append([0, 0])
        for action_type_id, action_name in enumerate(self._genv.action_names):
            action_mapping = self._genv.action_input_mappings[action_name]
            input_mappings = action_mapping["InputMappings"]

            for action_id in range(1, len(input_mappings) + 1):
                mapping = input_mappings[str(action_id)]
                description = mapping["Description"]
                actions.append(description)

                flat_action_mapping.append([action_type_id, action_id])

        if self._genv.player_count > 1:
            action_space = MultiAgentActionSpace(
                [gym.spaces.Discrete(len(flat_action_mapping)) for _ in range(self._genv.player_count)])
        else:
            action_space = gym.spaces.Discrete(len(flat_action_mapping))

        return action_space, flat_action_mapping

    def step(self, action):
        if self._genv.player_count > 1:
            g_action = [self.flat_action_mapping[a] for a in action]
        else:
            g_action = self.flat_action_mapping[action]
        observation, reward, done, info = self.env.step(g_action)
        reward /= 10
        inventory = self.game.get_global_variable([
            "inv_wood_sword", "inv_stone_sword", "inv_iron_sword", "inv_wood_pickaxe", "inv_stone_pickaxe",
            "inv_iron_pickaxe", "inv_sapling", "inv_stone", "inv_coal", "inv_wood", "inv_iron", "inv_diamond",
            "inv_food", "inv_drink", "inv_energy", "health",
        ])
        achievements = self.game.get_global_variable([
            "ach_collect_coal", "ach_collect_diamond", "ach_collect_drink", "ach_collect_iron", "ach_collect_sapling",
            "ach_collect_stone", "ach_collect_wood", "ach_defeat_skeleton", "ach_defeat_zombie", "ach_defeat_player",
            "ach_eat_cow", "ach_eat_plant", "ach_make_iron_pickaxe", "ach_make_iron_sword", "ach_make_stone_pickaxe",
            "ach_make_stone_sword", "ach_make_wood_pickaxe", "ach_make_wood_sword", "ach_place_furnace",
            "ach_place_plant", "ach_place_stone", "ach_place_table", "ach_wake_up"
        ])
        info['inventory'] = {key if not key.startswith('inv_') else key[4:]: value[1] for key, value in inventory.items()}
        info['achievements'] = {key[4:]: value[1] for key, value in achievements.items()}
        info['reward'] = reward
        info['discount'] = int(not done)

        return self._process_observation(observation), reward, done, info

    def reset(self):
        if self._level_id is None:
            level_string = self._generator.generate()
            reset_obs = self.env.reset(level_string=level_string)
        else:
            reset_obs = self.env.reset(level_id=self._level_id)

        return self._process_observation(reset_obs)

    def render(self, size=None, mode="rgb_array", observer=0):
        frame = self.env.render(mode=mode, observer=observer)
        if size:
            frame = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_NEAREST_EXACT)
        return frame
