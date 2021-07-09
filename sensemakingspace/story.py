
from .state import State
import pandas as pd
import numpy as np
import os

class Story:

    def __init__(self, user_id, task_id, data):

        self.user = user_id
        self.task = task_id

        self.accuracy = data['accuracy']
        self.autocomplete = data['autoCompleteUsed']
        self.prediction_rank = data['rankOfPredictionUsed']
        self.difficulty = data['difficulty']
        self.supported = data['supported']
        self.training = data['training']

        self.is_gt = False

        self.dataset = pd.read_csv('datasets/{}.csv'.format(data['dataset']))

        self.states = [State(st['timestamp'], st['selection'], st['turnedPrediction']) for st in data['selectionSequence']]

    def __len__(self):
        return len(self.states)

    def __str__(self):
        return 'Story(\n\t<{length} State(s)>,\n\tuser="{user}",\n\ttask="{task}"\n)'.format(
            length=len(self),
            user=self.user,
            task=self.task
        )

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):
        return self.states[key]

    def normalizedCoords(self):
        x = np.array(self.dataset['X'])
        y = np.array(self.dataset['Y'])

        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min)

        y_min = y.min()
        y_max = y.max()
        y = (y - y_min) / (y_max - y_min)

        x_norm_sel = []
        y_norm_sel = []

        for st in self.states:
            x_norm_sel.append(x[st.selection])
            y_norm_sel.append(y[st.selection])

        return x_norm_sel, y_norm_sel

    def encode(self, num_bins=10):
        histograms = []
        for x, y in zip(*self.normalizedCoords()):
            hist, _, _ = np.histogram2d(x, y, num_bins)
            histograms.append(hist)

        return np.stack(histograms)

    def change_string_list(self):
        assert len(self) > 2, 'Story must have at least two states!'
        strings = ['None']
        for idx, state in enumerate(self.states[:-1]):
            strings.append(state.change_string(self.states[idx+1]))
        return strings

    def print_changes(self):
        for s in self.change_string_list()[1:]:
            print(s)
