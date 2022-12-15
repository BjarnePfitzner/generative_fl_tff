import attr
import os
import logging

from tensorflow_federated.python.simulation import FileCheckpointManager


@attr.s(eq=False, frozen=True)
class TrainingState(object):
    server_state = attr.ib()
    client_states = attr.ib()


def get_write_model_checkpoint_fn(run_dir, disable_checkpointing=False):
    if disable_checkpointing:
        # This ugliness is necessary to bypass error described in comment below
        return (lambda ss, cs: (ss, cs, 1)), (lambda *args: None)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    checkpoint_manager = FileCheckpointManager(checkpoint_dir)

    def get_initial_state(initial_server_state, initial_client_states):
        initial_state = TrainingState(initial_server_state, initial_client_states)
        # Todo: checkpointing doesn't work for sync_d due to client_states holding weights / optimizer_state that don't
        #       support 'tf.convert_to_tensor'.
        training_state, checkpoint_round_num = checkpoint_manager.load_latest_checkpoint_or_default(initial_state)
        if checkpoint_round_num == 0:
            logging.info('No Checkpoint loaded')
            start_round = 1
        else:
            logging.info(f"Loaded Checkpoint for round {checkpoint_round_num}.")
            start_round = checkpoint_round_num + 1
        return training_state.server_state, training_state.client_states, start_round

    def write_model_checkpoint(round_num, server_state, client_states):
        state = TrainingState(server_state, client_states)
        try:
            logging.info(f'Saving checkpoint for round {round_num}')
            checkpoint_manager.save_checkpoint(state, round_num)
        except Exception as e:
            logging.info(f'Checkpoint saving exception: {e}')

    return get_initial_state, write_model_checkpoint
