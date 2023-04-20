import functools

import torch
import logging
from ..pattern_matcher import (
    Arg,
    CallFunction,
    get_arg_value,
    MULTIPLE,
    PatternEntry,
)
from torch._dynamo.utils import counters

log = logging.getLogger(__name__)

# Normalize split with split_size into split_with_sizes, so that we only deal with one type of split in
# subsequent optimizations
class NormalizeSplit(PatternEntry):
    def apply(self, match, graph, node):
        split_node = match.nodes[0]
        split_dim = get_arg_value(split_node, 2, "dim")
        split_size = get_arg_value(split_node, 1, "split_size_or_sections")
        if isinstance(split_size, list):
            return
        if "tensor_meta" not in split_node.meta:
            log.warning("tensor_meta absent for node", split_node)
            return
        split_sections = [
            meta.shape[split_dim] for meta in split_node.meta["tensor_meta"]
        ]
        split_input = split_node.args[0]
        with graph.inserting_after(split_node):
            new_split_node = graph.call_function(
                torch.split, args=(split_input, split_sections, split_dim)
            )
        split_node.replace_all_uses_with(new_split_node)
        graph.erase_node(split_node)
        counters["inductor"]["split_cat_norm"] += 1


@functools.lru_cache(None)
def _split_cat_init():
    from .pre_grad import patterns

    for pattern in [
        CallFunction(torch.split, Arg(), Arg(), Arg(), _users=MULTIPLE),
        CallFunction(torch.split, Arg(), Arg(), dim=Arg(), _users=MULTIPLE),
        CallFunction(
            torch.split, Arg(), split_size_or_sections=Arg(), dim=Arg(), _users=MULTIPLE
        ),
    ]:
        pattern = NormalizeSplit(pattern=pattern, extra_check=lambda arg: True)
        pattern.register(patterns)
