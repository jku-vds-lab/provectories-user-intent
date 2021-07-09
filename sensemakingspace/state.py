
import numpy as np

class State:
    
    def __init__(self, timestamp, selection, turned_pred):
        self.timestamp = timestamp
        # selection is a list of indices
        self.selection = selection
        self.turned_pred = turned_pred

    def __str__(self):
        pstr = 'State(\n'
        pstr += '\t"timestamp": {},\n'.format(self.timestamp)
        pstr += '\t"selection": {},\n'.format(self.selection)
        pstr += '\t"turnedPrediction": {},\n'.format(self.turned_pred)
        pstr += '\n)'
        return pstr

    def __repr__(self):
        return 'State(<{} point(s) selected>)'.format(len(self.selection))

    def change_string(self, other):
        if self.selection != other.selection:
            removed = set(self.selection) - set(other.selection)
            added = set(other.selection) - set(self.selection)
            if len(removed) != 0 and len(added) == 0:
                change = 'Removed {removed} from selection.'.format(
                    removed=removed
                )
            elif len(removed) == 0 and len(added) != 0:
                change = 'Added {added} to selection.'.format(
                    added=added
                )
            elif len(removed) != 0 and len(added) != 0:
                change = 'Added {added} to selection and removed {removed} from selection'.format(
                    added=added,
                    removed=removed
                )
            return change
        else:
            return 'No changes.'