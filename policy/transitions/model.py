from policy.transitions.extensions.markup import MarkupMachine
from policy.transitions import Machine
import json

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

