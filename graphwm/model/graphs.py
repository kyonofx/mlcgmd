"""
adapted from https://github.com/deepmind/graph_nets.
"""
import collections

NODES = "nodes"
EDGES = "edges"
EDGE_TYPE = "edge_type"
RECEIVERS = "receivers"
SENDERS = "senders"
GLOBALS = "globals"
N_NODE = "n_node"
N_EDGE = "n_edge"
COORDS = "coords"

GRAPH_FEATURE_FIELDS = (NODES, EDGES, COORDS, GLOBALS)
GRAPH_INDEX_FIELDS = (RECEIVERS, SENDERS)
GRAPH_DATA_FIELDS = (NODES, EDGES, COORDS, EDGE_TYPE, RECEIVERS, SENDERS, GLOBALS)
GRAPH_NUMBER_FIELDS = (N_NODE,)
ALL_FIELDS = (NODES, EDGES, COORDS, RECEIVERS, SENDERS, GLOBALS, N_NODE)


class GraphsTuple(
    collections.namedtuple("GraphsTuple",
                           GRAPH_DATA_FIELDS + GRAPH_NUMBER_FIELDS)):
  """Default namedtuple describing `Graphs`s.

  A children of `collections.namedtuple`s, which allows it to be directly input
  and output from `tensorflow.Session.run()` calls.

  An instance of this class can be constructed as
  ```
  GraphsTuple(nodes=nodes,
              edges=edges,
              globals=globals,
              receivers=receivers,
              senders=senders,
              n_node=n_node)
  ```
  where `nodes`, `edges`, `globals`, `receivers`, `senders`, `n_node` and
  `n_edge`(not needed) are arbitrary, but are typically numpy arrays, tensors, or `None`;
  see module's documentation for a more detailed description of which fields
  can be left `None`.
  """

  def _validate_none_fields(self):
    """Asserts that the set of `None` fields in the instance is valid."""
    if self.n_node is None:
      raise ValueError("Field `n_node` cannot be None")
    if self.receivers is None and self.senders is not None:
      raise ValueError(
          "Field `senders` must be None as field `receivers` is None")
    if self.senders is None and self.receivers is not None:
      raise ValueError(
          "Field `receivers` must be None as field `senders` is None")
    if self.receivers is None and self.edges is not None:
      raise ValueError(
          "Field `edges` must be None as field `receivers` and `senders` are "
          "None")

  def __init__(self, *args, **kwargs):
    del args, kwargs
    # The fields of a `namedtuple` are filled in the `__new__` method.
    # `__init__` does not accept parameters.
    super(GraphsTuple, self).__init__()
    self._validate_none_fields()

  def replace(self, **kwargs):
    output = self._replace(**kwargs)
    output._validate_none_fields()  # pylint: disable=protected-access
    return output

  def map(self, field_fn, fields=GRAPH_FEATURE_FIELDS):
    """Applies `field_fn` to the fields `fields` of the instance.

    `field_fn` is applied exactly once per field in `fields`. The result must
    satisfy the `GraphsTuple` requirement w.r.t. `None` fields, i.e. the
    `SENDERS` cannot be `None` if the `EDGES` or `RECEIVERS` are not `None`,
    etc.

    Args:
      field_fn: A callable that take a single argument.
      fields: (iterable of `str`). An iterable of the fields to apply
        `field_fn` to.

    Returns:
      A copy of the instance, with the fields in `fields` replaced by the result
      of applying `field_fn` to them.
    """
    return self.replace(**{k: field_fn(getattr(self, k)) for k in fields})