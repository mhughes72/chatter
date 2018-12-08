import copy
from abc import ABCMeta, abstractmethod

class Chain():
    def __init__(self):
        self.processors = []

    def append(self, processor, config=None):
        self.processors.append(processor(config))

    def _verify(self, errors):
        check_val = True in errors
        return check_val

    def run(self, items):
        processed_data = [ ]

        for item in items:
            temp_item = copy.deepcopy(item)

            errors = [ ]
            for processor in self.processors:
                temp_item, error = processor.run(temp_item)
                errors.append(error)

            error_status = self._verify(errors)
            if not error_status:
                processed_data.append(temp_item)

        return processed_data



class BaseProcessor( ):
    __metaclass__ = ABCMeta
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self, item):
        """
        Return value is the processed item.
        """
        # return processed_item, error status True = bad False/None = good
        return item, False