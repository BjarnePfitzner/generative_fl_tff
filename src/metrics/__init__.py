def get_metrics_list(cfg):
    columns = ['train_loss']
    if cfg.model.type != 'centralized':
        columns += ['n_sampled_clients']
    if cfg.evaluation.plot_l2_norms:
        columns += ['L2-Norms/mean_client_delta', 'L2-Norms/median_client_delta', 'L2-Norms/client_delta_aggregate']
    if cfg.evaluation.run_cls_eval:
        columns += ['classification/gen_acc', 'classification/real_acc', 'classification/acc_diff']
    if cfg.evaluation.run_fid_eval:
        columns += ['fid/fid']
    if cfg.differential_privacy.type != 'disabled':
        columns += ['RDP/epsilon', 'RDP/opt_order']
    return columns
