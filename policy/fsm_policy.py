from __future__ import annotations
import logging

from typing import Optional, Any, Dict, List, Text
from pathlib import Path

import rasa.utils.io
import rasa.shared.utils.io
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import State, Domain
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.featurizers.tracker_featurizers import FEATURIZER_FILE
from rasa.shared.exceptions import FileIOException
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.core.policies.policy import PolicyPrediction, Policy
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.structures import StoryStep
from rasa.core.constants import (
    DEFAULT_MAX_HISTORY,
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
)
from rasa.shared.core.events import (
    UserUttered,
    ActionExecuted
)

from policy.transitions.extensions.markup import MarkupMachine
from policy.transitions import Machine
logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=True
)
class CustomPolicy(Policy):
    """基于有限状态机来做的自定义对话响应策略.
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        return {
            "enable_feature_string_compression": True,
            "use_nlu_confidence_as_score": False,
            POLICY_PRIORITY: 4,
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
        fsm_infos: Optional[Dict] = None,
    ) -> None:
        """Initialize the policy."""
        super().__init__(config, model_storage, resource, execution_context, featurizer)
        self.fsm_infos = fsm_infos or {}

    #确保选择user_action,并且不是rulepolicy的输出action
    def validate_action_name(self, action_name) -> Optional[bool]:
        return action_name.startswith("utter_") and not action_name.startswith("utter_faq") and not action_name.startswith("utter_rule")
    #确保fsm预测和训练时忽略初始intent-开始，忽略rule管理的faq
    def validate_intent_name(self, intent_name) -> Optional[bool]:
        return intent_name != '开始' and not intent_name.startswith("faq")

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:


        file_importer = TrainingDataImporter.load_from_config(
                'config.yml', 'domain.yml', ['data']
            )
        #story中不要用or关键词，否则生成event的时候，会出现以intent结尾
        stories = file_importer.get_stories()

        fsm_states = []
        transitions = []
        for fsm_state in domain.user_actions:
            if self.validate_action_name(fsm_state):
                fsm_states.append(fsm_state)

        for story in stories.story_steps:
            actions = []
            intents = []
            if isinstance(story, StoryStep):
                for i in story.events:
                    if isinstance(i, UserUttered):
                        # 剔除默认开始inten和faq , 开始不会引起任何状态变化，因为状态机初始状态为utter_开场白，faq受rule管理
                        if self.validate_intent_name(i.intent.get("name")):
                            intents.append(i.intent.get("name"))
                    elif isinstance(i, ActionExecuted):
                        if self.validate_action_name(i.action_name):
                            actions.append(i.action_name)
                    else:
                        pass

            j = 0
            for i in range(0, len(actions) - 1):
                if j < len(intents):
                    #fsm_machine.add_transition(intents[j], fsm_states[i], states[i + 1])
                    temp_transition = [intents[j], actions[i], actions[i + 1]]
                    if temp_transition not in transitions:
                        transitions.append(temp_transition)
                j += 1

        fsm_machine = Model(name="bot1", states=fsm_states,transitions=transitions, initial="utter_开场白")

        self.fsm_infos = fsm_machine.to_json()

        logger.debug(f"FSM builded with {len(self.fsm_infos.get('states'))} unique states and  "
                     f"{len(self.fsm_infos.get('transitions'))} transitions")
        #保存为json
        self.persist()
        return self._resource


    def get_current_intent(self, tracker: DialogueStateTracker) -> Optional[Text]:
        return tracker.latest_message.intent['name']

    def get_last_action(self, tracker: DialogueStateTracker) -> Optional[Text]:
        latest_action_name = None
        #同步fsm预测时间为action_listen动作之后
        if tracker.latest_action_name != "action_listen":
            return None
        events = tracker.events.copy()
        while(len(events) > 0):
            event = events.pop()
            if isinstance(event, ActionExecuted):
                if self.validate_action_name(event.action_name):
                    latest_action_name = event.action_name
                    return latest_action_name

        return latest_action_name


    def _recall_states(self, states: List[State], tracker: DialogueStateTracker) -> Optional[Text]:

        current_intent_name = tracker.latest_message.intent['name']
        if not self.validate_intent_name(current_intent_name):
            return None

        last_action_name = self.get_last_action(tracker)
        if last_action_name is not None:
            logger.debug(f"Get valid user action as FSM state '{last_action_name}'")
            fsm = RecoveredModel(self.fsm_infos)
            fsm.to_state(last_action_name)
            logger.debug(f"FSM started, current state is '{last_action_name}'")

            try:
                fsm.model.trigger(current_intent_name)
            except:
                logger.debug("Trigger is invaild, fsm no predicted action!")
                return None
            logger.debug(f"Trigger {current_intent_name}  excuted, FSM state from '{last_action_name}' has changed to {fsm.model.state} ")
            return fsm.model.state
        else:
            logger.debug("There is system action action_listen from FSM")
            return "action_listen"

    def recall(
        self,
        states: List[State],
        tracker: DialogueStateTracker
    ) -> Optional[Text]:

        return self._recall_states(states, tracker)

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
        """根据tracker中的上下文预测机器人的下个动作.

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

        states = self._prediction_states(tracker, domain, rule_only_data=rule_only_data)
        logger.debug(f"Current tracker state:{self.format_tracker_states(states)}")

        #预测动作
        predicted_action_name = self.recall(
            states, tracker, domain, rule_only_data=rule_only_data
        )
        if predicted_action_name is not None:
            logger.debug(f"There is a fsm next action '{predicted_action_name}'")
            result = self._prediction_result(predicted_action_name, tracker, domain)
        else:
            logger.debug("There is no fsm next action")

        return self._prediction(result)

    def _metadata(self) -> Dict[Text, Any]:
        return self.fsm_infos

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "fsm_turns.json"

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
    ) -> CustomPolicy:
        """Loads a trained policy (see parent class for full docstring)."""
        featurizer = None
        #fsm_infos = None

        try:
            with model_storage.read_from(resource) as path:
                metadata_file = Path(path) / cls._metadata_filename()
                metadata = rasa.shared.utils.io.read_json_file(metadata_file)
                #fsm_infos = metadata

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
            fsm_infos=metadata,
        )



class Model(object):

    # Define some states. Most of the time, narcoleptic superheroes are just like
    # everyone else. Except for...
    def __init__(self, name, states, transitions, initial):
        self.name = name

        self.states = states

        self.transitions = transitions

        self.initial = initial

        # Initialize the state machine
        self.machine = MarkupMachine(model=self, states=self.states, transitions=self.transitions, initial=self.initial)


    def to_json(self):
        return self.machine.markup

class RecoveredModel(object):

    # Define some states. Most of the time, narcoleptic superheroes are just like
    # everyone else. Except for...
    def __init__(self, config, **kwargs):
        self.model = Machine(before_state_change=config['before_state_change'],
                             after_state_change=config['after_state_change'],
                             prepare_event=config['prepare_event'],
                             finalize_event=config['finalize_event'],
                             send_event=config['send_event'],
                             auto_transitions=config['send_event'],
                             ignore_invalid_triggers=config['send_event'],
                             queued=config['queued'],
                             initial=config['initial'],
                             transitions=config['transitions'],
                             states=config['states'])

    def to_state(self, state):
        self.model.set_state(state)

