import attr
import tensorflow as tf

from src.utils.tf_utils import get_weights
from src.utils.optimizer_utils import get_optimizer_state, initialize_optimizer_vars


@attr.s(eq=False, frozen=True)
class ClientOutput(object):
    """Structure for outputs returned from clients during federated optimization.

    Fields:
    -   `weights_delta`: A dictionary of updates to the model's trainable
        variables.
    -   `client_weight`: Weight to be used in a weighted mean when
        aggregating `weights_delta`.
    -   `model_output`: A structure matching `tff.learning.Model.report_local_outputs`, reflecting the results of
       training on the input dataset.
    -   `client_optimizer_states`: The optimizer variables after training.
    -   `client_state`: The updated `ClientState`.
  """
    weights_delta = attr.ib()
    client_weight = attr.ib()

    train_metrics = attr.ib()
    client_state = attr.ib()


@attr.s(eq=False, frozen=False)
class ServerState(object):
    """Structure for state on the server.
    Fields:
    -   `model_weights`: A dictionary of model's trainable variables.
    -   `aggregation_optimizer_state`: Variables of server optimizer.
    -   `aggregation_state`: State of differential privacy aggregator.
    -   'counters': Counters such as total iters count
    """
    model_weights = attr.ib()
    aggregation_optimizer_state = attr.ib()
    counters = attr.ib()
    aggregation_state = attr.ib(default=())


@attr.s(eq=False, frozen=True, slots=True)
class ServerMessage(object):
    """Structure for tensors broadcasted by server during federated optimization.
    Fields:
    -   `model_weights`: A dictionary of model's trainable tensors.
    -   `client_optimizer_state`: State of the aggregated client optimizer.
    """
    model_weights = attr.ib()
    global_round = attr.ib()


def server_initial_state(model, aggregation_optimizer, aggregation_state=()):
    initialize_optimizer_vars(aggregation_optimizer, model)

    return ServerState(
        model_weights=get_weights(model),
        aggregation_optimizer_state=get_optimizer_state(aggregation_optimizer),
        counters={
            'num_rounds': tf.constant(0),
            'total_iters_count': tf.constant(0)
        },
        aggregation_state=aggregation_state)


@tf.function
def server_computation(server_state: ServerState,
                       model: tf.keras.Model,
                       aggregation_optimizer: tf.optimizers.Optimizer,
                       new_aggregation_state,
                       weights_delta,
                       total_iters_count) -> ServerState:
    """Updates `server_state` based on `weights_delta`.
    Args:
      server_state: A `ServerState`, the state to be updated.
      model: A `KerasModelWrapper` or `tff.learning.Model`.
      aggregation_optimizer: A `tf.keras.optimizers.Optimizer`. If the optimizer
        creates variables, they must have already been created.
      new_aggregation_state: New aggregation state
      # new_client_optimizer_states: New client optimizer state
      weights_delta: A nested structure of tensors holding the updates to the
        trainable variables of the model.
      total_iters_count: A scalar to update `ServerState.total_iters_count`.
    Returns:
      An updated `ServerState`.
    """
    # A tf.function can't modify the structure of its input arguments,
    # so we make a semi-shallow copy:
    server_state = attr.evolve(server_state, counters=dict(server_state.counters))

    # Initialize the model with the current state.
    tf.nest.map_structure(lambda a, b: a.assign(b), model.weights, server_state.model_weights)
    tf.nest.assert_same_structure(weights_delta, model.weights)
    tf.nest.map_structure(lambda x: tf.debugging.assert_all_finite(x, f'aggregated weight updates include NaN or Inf: {x}'), weights_delta)

    aggregation_optimizer_state = get_optimizer_state(aggregation_optimizer)
    tf.nest.map_structure(lambda a, b: a.assign(b), aggregation_optimizer_state, server_state.aggregation_optimizer_state)

    # Apply the update to the model.
    grads_and_vars = tf.nest.map_structure(lambda x, v: (-1.0 * x, v), weights_delta,
                                           model.weights)
    aggregation_optimizer.apply_gradients(grads_and_vars, name='server_update')

    # Update the state of the DP averaging aggregator
    server_state.aggregation_state = new_aggregation_state

    server_state.model_weights = get_weights(model)
    server_state.aggregation_optimizer_state = aggregation_optimizer_state
    server_state.counters['num_rounds'] += 1
    server_state.counters['total_iters_count'] = total_iters_count

    return server_state
