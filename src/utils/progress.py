class Progress:
    def __init__(self, total_rounds, rounds_per_eval):
        self.total_rounds = total_rounds
        self.rounds_per_eval = rounds_per_eval

        self._total_train_duration = 0
        self._total_eval_duration = 0

    def eta(self, epoch_num):
        remaining_number_evals = (self.total_rounds - epoch_num) // self.rounds_per_eval + 1
        number_evals_so_far = epoch_num // self.rounds_per_eval
        if number_evals_so_far == 0:
            remaining_eval_duration = remaining_number_evals * 100
        else:
            remaining_eval_duration = self._total_eval_duration / number_evals_so_far * remaining_number_evals
        remaining_train_duration = self._total_train_duration / epoch_num * (self.total_rounds - epoch_num)
        remaining_runtime = remaining_eval_duration + remaining_train_duration

        hours = int(remaining_runtime // 3600)
        minutes = int(remaining_runtime % 3600 // 60)

        return hours, minutes

    def add_train_duration(self, duration):
        self._total_train_duration += duration

    def add_eval_duration(self, duration):
        self._total_eval_duration += duration
