from typing import List
from utils.instance import Instance, InstanceList

from .text_sum_dataset import TextSumDataset

def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)
