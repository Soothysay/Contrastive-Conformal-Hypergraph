import numpy as np
import torch
import torch.nn.functional as F


def _check_quantile_level(q_level):
    """Fail early when a calibration split would produce an invalid quantile."""
    if not 0 <= q_level <= 1:
        raise ValueError(
            "Invalid conformal quantile level. Increase the calibration split "
            "size or choose a larger alpha."
        )


def tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    """Top-probability conformal score: include classes above a probability cut."""
    cal_scores = 1-cal_smx[np.arange(n),cal_labels]
    q_level = np.ceil((n+1)*(1-alpha))/n
    _check_quantile_level(q_level)
    qhat = np.quantile(cal_scores, q_level, method='higher')
    prediction_sets = val_smx >= (1-qhat)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff

def aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    """Adaptive prediction sets based on cumulative sorted probabilities."""
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    _check_quantile_level(q_level)
    qhat = np.quantile(
        cal_scores, q_level, method="higher"
    )
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff

def raps(
    cal_smx,
    val_smx,
    cal_labels,
    val_labels,
    n,
    alpha,
    lam_reg=0.01,
    k_reg=5,
    disallow_zero_sets=False,
    rand=True,
):
    """Regularized APS; penalizes labels after the top-k classes."""
    k_reg = min(k_reg, cal_smx.shape[1])
    reg_vec = np.array(k_reg*[0,] + (cal_smx.shape[1]-k_reg)*[lam_reg,])[None,:]

    cal_pi = cal_smx.argsort(1)[:,::-1];
    cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]
    # Get the score quantile
    q_level = np.ceil((n+1)*(1-alpha))/n
    _check_quantile_level(q_level)
    qhat = np.quantile(cal_scores, q_level, method='higher')
    # Deploy
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
def threshold(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    """Cumulative-probability threshold baseline with non-empty set guard."""
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]

    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)

    prediction_sets = np.take_along_axis(val_srt <= 1-alpha, val_pi.argsort(axis=1), axis=1)
    prediction_sets[np.arange(prediction_sets.shape[0]), val_pi[:, 0]] = True

    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
def run_conformal_classification(pred, data, n, alpha, score = 'aps',
                                 calib_eval = False, validation_set = False,
                                 use_additional_calib = False, return_prediction_sets = False, calib_fraction = 0.5,
                                 conformal_trials=100, raps_lam_reg=0.01, raps_k=5):
    """Repeatedly split calibration/evaluation masks and report coverage/size."""
    if calib_eval:
        n_base = int(n * (1-calib_fraction))
    else:
        n_base = n

    logits = torch.nn.Softmax(dim = 1)(pred).detach().cpu().numpy()
    calib_test_mask = data.calib_test_mask.detach().cpu().numpy()
    if validation_set:
        smx = logits[data.valid_mask.detach().cpu().numpy()]
        labels = data.y[data.valid_mask].detach().cpu().numpy()
        n_base = int(len(np.where(data.valid_mask.detach().cpu().numpy())[0])/2)
    else:
        if calib_eval:
            smx = logits[data.calib_test_real_mask.detach().cpu().numpy()]
            labels = data.y[data.calib_test_real_mask].detach().cpu().numpy()
        else:
            smx = logits[calib_test_mask]
            labels = data.y[data.calib_test_mask].detach().cpu().numpy()

    cov_all = []
    eff_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []

    for k in range(conformal_trials):
        idx = np.array([1] * n_base + [0] * (smx.shape[0]-n_base)) > 0
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_smx, val_smx = smx[idx,:], smx[~idx,:]
        cal_labels, val_labels = labels[idx], labels[~idx]

        if use_additional_calib and calib_eval:
            smx_add = logits[data.calib_eval_mask]
            labels_add = data.y[data.calib_eval_mask].detach().cpu().numpy()
            cal_smx = np.concatenate((cal_smx, smx_add))
            cal_labels = np.concatenate((cal_labels, labels_add))

        n = cal_smx.shape[0]

        if score == 'tps':
            prediction_sets, cov, eff = tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)
        elif score == 'aps':
            prediction_sets, cov, eff = aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)
        elif score == 'raps':
            prediction_sets, cov, eff = raps(
                cal_smx,
                val_smx,
                cal_labels,
                val_labels,
                n,
                alpha,
                lam_reg=raps_lam_reg,
                k_reg=raps_k,
            )
        elif score == 'threshold':
            prediction_sets, cov, eff = threshold(cal_smx, val_smx, cal_labels, val_labels, n, alpha)
        else:
            raise ValueError(f"Unknown conformal score: {score}")

        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)

    if return_prediction_sets:
        return cov_all, eff_all, pred_set_all, val_labels_all, idx_all
    else:
        return np.mean(cov_all), np.mean(eff_all)

def sim(z1: torch.Tensor, z2: torch.Tensor):
    """Cosine-similarity matrix between two batches of embeddings."""
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def InfoNCE(z1, z2, T=0.2):
    """Symmetric InfoNCE loss for two augmented embedding views."""
    l1 = semi_loss(z1, z2, T)
    l2 = semi_loss(z2, z1, T)

    ret = (l1 + l2) * 0.5
    ret = ret.mean()

    return ret


def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T):
    """One direction of the contrastive loss used by InfoNCE."""
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
