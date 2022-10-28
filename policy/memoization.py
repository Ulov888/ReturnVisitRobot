from __future__ import annotations
import zlib

import base64
import json
import logging

from tqdm import tqdm
from typing import Optional, Any, Dict, List, Text
from pathlib import Path

import rasa.utils.io
import rasa.shared.utils.io
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import State, Domain
from rasa.shared.core.events import ActionExecuted
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.featurizers.tracker_featurizers import FEATURIZER_FILE
from rasa.shared.exceptions import FileIOException
from rasa.core.policies.policy import PolicyPrediction, Policy, SupportedData
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates, TrainingDataGenerator
from rasa.shared.utils.io import is_logging_disabled
from rasa.core.constants import (
    MEMOIZATION_POLICY_PRIORITY,
    DEFAULT_MAX_HISTORY,
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.slots import CategoricalSlot
from rasa.shared.core.events import SlotSet, ActiveLoop

from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.core.training_data.structures import StoryGraph, StoryStep

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=True
)
class MemoizationPolicy(Policy):
    """A policy that follows exact examples of `max_history` turns in training stories.

    Since `slots` that are set some time in the past are
    preserved in all future feature vectors until they are set
    to None, this policy implicitly remembers and most importantly
    recalls examples in the context of the current dialogue
    longer than `max_history`.

    This policy is not supposed to be the only policy in an ensemble,
    it is optimized for precision and not recall.
    It should get a 100% precision because it emits probabilities of 1.1
    along it's predictions, which makes every mistake fatal as
    no other policy can overrule it.

    If it is needed to recall turns from training dialogues where
    some slots might not be set during prediction time, and there are
    training stories for this, use AugmentedMemoizationPolicy.
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            "enable_feature_string_compression": True,
            "use_nlu_confidence_as_score": False,
            POLICY_PRIORITY: MEMOIZATION_POLICY_PRIORITY,
            POLICY_MAX_HISTORY: DEFAULT_MAX_HISTORY,
        }

    def _standard_featurizer(self) -> MaxHistoryTrackerFeaturizer:
        # Memoization policy always uses MaxHistoryTrackerFeaturizer
        # without state_featurizer
        return MaxHistoryTrackerFeaturizer(
            state_featurizer=None, max_history=self.config[POLICY_MAX_HISTORY]
        )

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional[TrackerFeaturizer] = None,
        lookup: Optional[Dict] = None,
    ) -> None:
        """Initialize the policy."""
        super().__init__(config, model_storage, resource, execution_context, featurizer)
        self.lookup = lookup or {}

    def _create_lookup_from_states(
        self,
        trackers_as_states: List[List[State]],
        trackers_as_actions: List[List[Text]],
    ) -> Dict[Text, Text]:
        """Creates lookup dictionary from the tracker represented as states.

        Args:
            trackers_as_states: representation of the trackers as a list of states
            trackers_as_actions: representation of the trackers as a list of actions

        Returns:
            lookup dictionary
        """
        lookup = {}

        if not trackers_as_states:
            return lookup

        assert len(trackers_as_actions[0]) == 1, (
            f"The second dimension of trackers_as_action should be 1, "
            f"instead of {len(trackers_as_actions[0])}"
        )

        ambiguous_feature_keys = set()

        pbar = tqdm(
            zip(trackers_as_states, trackers_as_actions),
            desc="Processed actions",
            disable=is_logging_disabled(),
        )
        for states, actions in pbar:
            action = actions[0]

            feature_key = self._create_feature_key(states)
            if not feature_key:
                continue

            if feature_key not in ambiguous_feature_keys:
                if feature_key in lookup.keys():
                    if lookup[feature_key] != action:
                        # delete contradicting example created by
                        # partial history augmentation from memory
                        ambiguous_feature_keys.add(feature_key)
                        del lookup[feature_key]
                else:
                    lookup[feature_key] = action
            pbar.set_postfix({"# examples": "{:d}".format(len(lookup))})

        return lookup

    def _create_feature_key(self, states: List[State]) -> Text:
        # we sort keys to make sure that the same states
        # represented as dictionaries have the same json strings
        # quotes are removed for aesthetic reasons
        feature_str = json.dumps(states, sort_keys=True).replace('"', "")
        if self.config["enable_feature_string_compression"]:
            compressed = zlib.compress(
                bytes(feature_str, rasa.shared.utils.io.DEFAULT_ENCODING)
            )
            a = base64.b64encode(compressed).decode(
                rasa.shared.utils.io.DEFAULT_ENCODING
            )
            return a
        else:
            return feature_str

    def get_slot_set_event_id(self, var_judge_slots, slots_names):
        for idx in range(len(var_judge_slots)):
            if var_judge_slots[idx].key in slots_names:
                return idx
        return -1

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:

        # categorical_slots = []
        # forms_with_categorical_slots = {}
        # if len(domain.slots) > 0:
        #     for slot in domain.slots:
        #         if isinstance(slot, CategoricalSlot):
        #             categorical_slots.append(slot.name)
        #             for mapping in slot.mappings:
        #                 for condition in mapping.get("conditions"):
        #                     if forms_with_categorical_slots.get(condition['active_loop']):
        #                         forms_with_categorical_slots[condition['active_loop']].add(slot.name)
        #                     else:
        #                         forms_with_categorical_slots[condition['active_loop']] = set()
        #                         forms_with_categorical_slots[condition['active_loop']].add(slot.name)
        # #story自加载
        # data_loader = RasaFileImporter(config_file="config.yml", domain_path="domain.yml", training_data_paths="data\\")
        # #获取story集合，如果有checkpoint，此时story之间还未合并
        # story_graph = data_loader.get_stories()
        #
        # # 获取slot变量判断事件列表，每个事件只能对应一次插入，插入后立即删除该事件
        # var_judge_slots = []
        # #返回完整story(处理checkpoints或者or之后)
        # for step in story_graph.ordered_steps():
        #     #忽略rulestep
        #     if isinstance(step, StoryStep):
        #         temp_events = step.events.copy()
        #         while temp_events:
        #             event = temp_events.pop(0)
        #             if isinstance(event, SlotSet):
        #                 if event.key in categorical_slots:
        #                     var_judge_slots.append(event)
        #
        # # 对每个完整的tracker进行槽位填充，每个tracker需要填槽的form数量和slot变量判断事件一致
        # #gg_story = story_graph.ordered_steps()
        # for step in story_graph.ordered_steps():
        #     #忽略ruleStep
        #     if isinstance(step, StoryStep):
        #         temp_events = step.events.copy()
        #         new_events = []
        #         while temp_events:
        #             event = temp_events.pop(0)
        #             #判断条件-如果是带有Categoral类型slot的form表单并且变量判断事件列表不为空
        #             if isinstance(event, ActiveLoop) and event.name in forms_with_categorical_slots \
        #                     and len(var_judge_slots) > 0:
        #                 #获取form对应的需要手动填充的slotSet事件id
        #                 idx = self.get_slot_set_event_id(var_judge_slots, forms_with_categorical_slots[event.name])
        #                 if idx >= 0:
        #                     new_events.append(var_judge_slots.pop(idx))
        #             new_events.append(event)
        #         step.events.clear()
        #         #用插入填槽事件的events更新整个step，方便下游生成想要的tracker
        #         for event in new_events:
        #             step.add_event(event)
        # #tracker生成类
        # td = TrainingDataGenerator(story_graph=story_graph,
        #                            domain=domain,
        #                            remove_duplicates=True,
        #                            augmentation_factor=0,
        #                            use_story_concatenation=True,
        #                            debug_plots=False)
        # #调用tracker生成函数 同时默认生成tracker对应的states
        # new_training_trackers = td.generate()
        #
        # new_training_trackers = [
        #     t
        #     for t in new_training_trackers
        #     if not hasattr(t, "is_augmented") or not t.is_augmented
        # ]
        # new_training_trackers = SupportedData.trackers_for_supported_data(
        #     self.supported_data(), new_training_trackers
        # )

        (
            trackers_as_states,
            trackers_as_actions,
        ) = self.featurizer.training_states_and_labels(training_trackers, domain)


        #把story转成states的长字符串(压缩格式)和action的映射字典
        self.lookup = self._create_lookup_from_states(
            trackers_as_states, trackers_as_actions
        )
        logger.debug(f"Memorized {len(self.lookup)} unique examples.")
        #保存为json
        self.persist()
        return self._resource


    def _recall_states(self, states: List[List[State]]) -> Optional[Text]:
        states_list = states[0].copy()
        _states = []
        last_intent = ""
        while len(states_list) > 0:
            state = states_list.pop(0)
            if not state:
                _states.append(state)
            elif state.get("active_loop"):
                if len(states_list) > 0:
                    if states_list[0].get("prev_action").get("action_name") == 'action_listen' and \
                        states_list[0].get("user").get("intent") == state.get("user").get("intent") and \
                            states_list[0].get("slots"):
                        state['slots'] = states_list[0].get("slots")
                        states_list.pop(0)
                    #处理any或者填槽失败slot为空的穿插
                    elif states_list[0].get("prev_action").get("action_name") == 'action_listen' and \
                        states_list[0].get("user").get("intent") == state.get("user").get("intent") and \
                            not states_list[0].get("slots"):
                        states_list.pop(0)
                state.pop("active_loop")
                _states.append(state)
            else:
                _states.append(state)

        return self.lookup.get(self._create_feature_key(_states))

    def recall(
        self,
        states: List[List[State]],
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]],
    ) -> Optional[Text]:
        """Finds the action based on the given states.

        Args:
            states: List of states.
            tracker: The tracker.
            domain: The Domain.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.

        Returns:
            The name of the action.
        """
        return self._recall_states(states)

    def _prediction_result(
        self, action_name: Text, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        result = self._default_predictions(domain)
        if action_name:
            if self.config["use_nlu_confidence_as_score"]:
                # the memoization will use the confidence of NLU on the latest
                # user message to set the confidence of the action
                score = tracker.latest_message.intent.get("confidence", 1.0)
            else:
                score = 1.0

            result[domain.index_for_action(action_name)] = score

        return result

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.

        Returns:
             The policy's prediction (e.g. the probabilities for the actions).
        """
        #返回action的数量长度的list [0...]
        result = self._default_predictions(domain)
        #states1 = domain.get_active_state(tracker)
        states_real = domain.states_for_tracker_history(tracker)
        states = self._prediction_states(tracker, domain, rule_only_data=rule_only_data)
        logger.debug(f"Current tracker state:{self.format_tracker_states(states)}")
        current_state = tracker.current_state()
        #domain.retrieval_intents
        #真正预测的一步
        predicted_action_name = self.recall(
            [states, states_real], tracker, domain, rule_only_data=rule_only_data
        )
        if predicted_action_name is not None:
            logger.debug(f"There is a memorised next action '{predicted_action_name}'")
            result = self._prediction_result(predicted_action_name, tracker, domain)
        else:
            logger.debug("There is no memorised next action")

        return self._prediction(result)

    def _metadata(self) -> Dict[Text, Any]:
        return {"lookup": self.lookup}

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "memorized_turns.json"

    def persist(self) -> None:
        """Persists the policy to storage."""
        with self._model_storage.write_to(self._resource) as path:
            # not all policies have a featurizer
            if self.featurizer is not None:
                self.featurizer.persist(path)

            file = Path(path) / self._metadata_filename()

            rasa.shared.utils.io.create_directory_for_file(file)
            rasa.shared.utils.io.dump_obj_as_json_to_file(file, self._metadata())

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> MemoizationPolicy:
        """Loads a trained policy (see parent class for full docstring)."""
        featurizer = None
        lookup = None

        try:
            with model_storage.read_from(resource) as path:
                metadata_file = Path(path) / cls._metadata_filename()
                metadata = rasa.shared.utils.io.read_json_file(metadata_file)
                lookup = metadata["lookup"]

                if (Path(path) / FEATURIZER_FILE).is_file():
                    featurizer = TrackerFeaturizer.load(path)

        except (ValueError, FileNotFoundError, FileIOException):
            logger.warning(
                f"Couldn't load metadata for policy '{cls.__name__}' as the persisted "
                f"metadata couldn't be loaded."
            )

        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            featurizer=featurizer,
            lookup=lookup,
        )


