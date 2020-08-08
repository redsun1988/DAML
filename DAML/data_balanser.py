import pandas as pd
from typing import List
import numpy as np
from collections import Counter

class DataBalanser:
    def __init__(self, data_source: pd.DataFrame) -> None:
        self._data_source = data_source
        self._stategy: str = "downsampling"
        self._target_field = ""
        self._data_fields = []
        self._balansed_source = None
    
    def options_changed(self):
        self._balansed_source = None

    @property
    def target_field(self) -> str:
        return self._target_field
        
    @target_field.setter
    def target_field(self, value) -> str:
        self._target_field = value
        self.options_changed()
    
    @property
    def stategy(self) -> str:
        return self._stategy
        
    @stategy.setter
    def stategy(self, value) -> str:
        self._stategy = value
        self.options_changed()

    @property
    def data_fields(self) -> List[str]:
        return self._data_fields
                
    @property
    def data_source(self) -> pd.DataFrame:
        return self._data_source

    @data_source.setter
    def data_source(self, value) -> pd.DataFrame:
        self._data_source = value
        self.options_changed()

    @property
    def data(self):
        return self.balansed_source[self.data_fields].values
    
    @property    
    def target(self):
        return self.balansed_source[self.target_field].values
    
    @property
    def balansed_source(self):
        if self._balansed_source is None:
            filtered_data_parts = []
            counter = Counter(self.data_source[self.target_field].values)
            un_targets = list(counter.keys())

            if self._stategy == "downsampling":
                min_samples_count = min(counter.values())
                for u_target in un_targets:
                    filtered_data_parts.append(
                        self.data_source[self.data_source[self.target_field] == u_target].sample(
                            min_samples_count))

            # Create cashe and remove .sample(frac=1)
            self._balansed_source = pd.concat(filtered_data_parts, ignore_index=True).reset_index(drop=True)
        return self._balansed_source