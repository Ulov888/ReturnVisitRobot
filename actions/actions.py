from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, UserUtteranceReverted, UserUttered, \
    ActionExecutionRejected
import logging

logger = logging.getLogger(__name__)

class ActionRepeat(Action):
    """在触发复述意图的情况下，对上一轮次机器人播报的话术进行重述"""

    def name(self) -> Text:
        return "action_sys_repeat"

    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        events = tracker.current_state()['events']

        utter_action_list = []
        # 只提取当前轮次的对话bot事件即可
        for index in range(len(events) - 3, 0, -1):
            tmp_event = events[index]
            if tmp_event['event'] == 'user_featurization':
                break
            else:
                if tmp_event['event'] == 'bot':
                    utter_action = tmp_event['metadata']['utter_action']
                    if "utter_sys_unknown" in utter_action:
                        continue
                    utter_action_list.append(utter_action)

        if len(utter_action_list) <= 0:
            dispatcher.utter_message(response="utter_sys_unknown1")

        else:
            utter_action_list = list(reversed(utter_action_list))
            for action_name in utter_action_list:
                dispatcher.utter_message(response=action_name)

        return [UserUtteranceReverted()]


class ActionGoBack(Action):
    """在多轮对话过程中，询问FAQ时，将话术拉回到多轮对话中"""

    def name(self) -> Text:
        return "action_sys_goback"

    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        current_state = tracker.current_state()

        #去除填槽情况
        active_loop = current_state['active_loop']
        if 'name' in active_loop:
            return []

        events = current_state['events']
        event_list = []
        for index in range(len(events) - 3, 0, -1):
            tmp_event = events[index]
            if tmp_event['event'] == 'user':
                intent_name = tmp_event['parse_data']['intent']['name']
                # 去除FAQ意图及系统内置意图
                if "intent" not in intent_name or "intent_sys" in intent_name:
                    event_list = []
                    continue

                if len(event_list) >= 1:
                    break

            else:
                if tmp_event['event'] == 'bot':
                    event_list.append(tmp_event)

        # 去除不存在流程的情况
        if len(event_list) <= 0:
            return []

        # 去除存在流程，但已经结束的情况
        utter_action_list = []
        flow_over_flag = False
        for bot_event in event_list:
            utter_action = bot_event['metadata']['utter_action']
            if "utter_sys_unknown" in utter_action:
                continue

            utter_action_list.append(utter_action)

            data = bot_event['data']
            if "custom" not in data:
                logger.warning("action_sys_goback, the event haven't custom, event:{}".format(bot_event))
                continue

            custom = bot_event['data']['custom']
            if custom is None:
                logger.warning("action_sys_goback, the event custom is none, event:{}".format(bot_event))
                continue

            if "flow_over" not in custom:
                continue

            flow_over = custom['flow_over']
            if flow_over:
                flow_over_flag = True

        if flow_over_flag or len(utter_action_list) <= 0:
            return []

        utter_action_list = list(reversed(utter_action_list))
        for action_name in utter_action_list:
            dispatcher.utter_message(response=action_name)

        return [UserUtteranceReverted()]


class ActionSessionStart(Action):
    """初始化对话时进行信息初始化，将metadata初始传递进行的变量信息赋值到相应的槽位上"""

    def name(self) -> Text:
        return "action_session_start"

    @staticmethod
    def _slot_set_events_from_tracker(
            tracker: "DialogueStateTracker",
    ) -> List["SlotSet"]:
        """Fetch SlotSet events from tracker and carry over key, value and metadata."""
        slots = []
        for event in tracker.applied_events():
            if "slot" in event['event']:
                slots.append(SlotSet(key=event['name'], value=event['value']))

        return slots

    @staticmethod
    def _fetch_slots(tracker: "DialogueStateTracker",
                     ) -> List["SlotSet"]:
        slots = []
        metadata = tracker.get_slot("session_started_metadata")

        # 对话初始化相关的变量均放在vars字段中
        if metadata is None or "vars" not in metadata:
            return slots

        vars = metadata['vars']
        if metadata:
            for key in vars.keys():
                value = vars[key]
                slots.append(SlotSet(key=key, value=value))
        return slots

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        _events = [SessionStarted()]

        if "session_config" in domain and "carry_over_slots_to_new_session" in domain['session_config']:
            if domain['session_config']['carry_over_slots_to_new_session']:
                _events.extend(self._slot_set_events_from_tracker(tracker))

        _events.extend(self._fetch_slots(tracker))

        _events.append(ActionExecuted("action_listen"))
        return _events


class ActionDefaultFallback(Action):
    """未识别处理：
        1、nlu_fallback未识别的情况；
        2、DM阈值偏低的情况；
        在未识别触发时，如果是处于多轮对话中，需要将对话拉回，而且在填槽情况下，如果填槽到第三次需要抛出填槽失败意图
    """

    def name(self) -> Text:
        return "action_default_fallback"

    def _count_unknown_num(self, events: "List[Event]"):
        """判断是连续第几次未识别"""

        unknown_count = 1
        unknow_flag = False  # 一轮对话中是否是未识别标识

        repeat_flag = False  # 一轮对话中是否存在复述标识

        flow_over_flag = False  # 最新一轮对话是否流程已经结束

        action_name_list = []
        input_channel = None  # 临时取一下input_channel，用于unknown bot事件
        model_id = None  # 临时提取model_id，用于构建unknown bot事件

        for index in range(len(events) - 1, 0, -1):
            tmp_event = events[index]

            # 判断当前对话轮次中是否是未识别
            if tmp_event['event'] == 'action':
                if tmp_event['name'] == self.name():
                    unknow_flag = True

                if tmp_event['name'] == "action_sys_repeat" or tmp_event['name'] == "utter_sys_welcome":
                    repeat_flag = True
                continue

            if tmp_event['event'] == 'user':
                input_channel = tmp_event['input_channel']

                # 排除掉复述的情况
                if repeat_flag:
                    repeat_flag = False
                    action_name_list = []
                    continue

                # 属于未识别第一次中断跳出
                if not unknow_flag:
                    break
                else:
                    unknown_count += 1
                    unknow_flag = False

            # 提取最近轮次回复
            if tmp_event['event'] == 'bot' and unknown_count <= 1:
                model_id = tmp_event['metadata']['model_id']

                utter_action = tmp_event['metadata']['utter_action']
                if "utter_faq" in utter_action or "utter_sys_unknown" in utter_action:
                    continue
                action_name_list.append(utter_action)

                # 流程结束不再做拉回处理
                data = tmp_event['data']

                if "custom" not in data:
                    logger.warning("action_default_fallback, the event haven't custom, event:{}".format(tmp_event))
                    continue

                custom = data['custom']
                if custom is None:
                    logger.warning("action_default_fallback, the event custom is none, event:{}".format(tmp_event))
                    continue

                if "flow_over" not in custom:
                    continue
                flow_over = custom['flow_over']
                if flow_over:
                    flow_over_flag = True

        if flow_over_flag:
            action_name_list = []

        return unknown_count, action_name_list, input_channel, model_id

    @staticmethod
    def _make_intent(input_channel: str, form_action_name: str):
        """构建3次填槽失败事件"""
        parse_data = {'intent': {'name': 'intent_sys_fill_slot_failure', 'confidence': 1.0}, 'entities': [],
                      'text': '/intent_sys_fill_slot_failure', 'metadata': {},
                      'intent_ranking': [{'name': 'intent_sys_fill_slot_failure', 'confidence': 1.0}]}
        user = UserUttered(text="/intent_sys_fill_slot_failure", parse_data=parse_data, input_channel=input_channel)

        action_execution_rejected = ActionExecutionRejected(action_name=form_action_name, policy="RulePolicy",
                                                            confidence=1.0)

        logger.debug("action_default_fallback，slot fill failure, intent: intent_sys_fill_slot_failure")

        return [user, action_execution_rejected]

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        current_state = tracker.current_state()

        events = current_state['events']

        # 判断当前对话是否在填槽
        active_loop = current_state['active_loop']
        active_loop_flag = False
        if 'name' in active_loop:
            events = events[:len(events) - 4]
            active_loop_flag = True
        else:
            events = events[:len(events) - 3]

        unknown_count, action_name_list, input_channel, model_id = self._count_unknown_num(events)

        action_name_list = list(reversed(action_name_list))

        # 填槽失败3次的情况
        logger.debug(f"action_default_fallback unknown number:'{unknown_count}'")

        if active_loop_flag and unknown_count >= 3:
            form_action_name = active_loop['name']
            return self._make_intent(input_channel, form_action_name)

        slot_sys_unknown_num = int(tracker.get_slot('slot_sys_unknown_num'))
        # 连续未识别次数在预期范围内
        if unknown_count <= slot_sys_unknown_num:
            utter_unknown_action_name = 'utter_sys_unknown{}'.format(unknown_count)
            dispatcher.utter_message(response=utter_unknown_action_name)

            for action_name in action_name_list:
                dispatcher.utter_message(response=action_name)

            return [UserUtteranceReverted()]

        else:
            utter_unknown_action_name = 'utter_sys_unknown{}'.format(slot_sys_unknown_num)
            dispatcher.utter_message(response=utter_unknown_action_name)

            # 3次填槽范围内
            if active_loop_flag:
                for action_name in action_name_list:
                    dispatcher.utter_message(response=action_name)

            return [UserUtteranceReverted()]



