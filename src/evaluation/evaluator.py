import torch
from src.evaluation.metrics import calculate_best_of_k
from src.data_pipeline.dataset import SocialDataset
 
 
class Evaluator:
    """
    Standard ETH-UCY benchmark evaluator.
 
    Model output contract
    ---------------------
    Every model evaluated here must return predictions as a FloatTensor of
    shape [K, N, pred_len, 2] in the NORMALISED absolute coordinate frame —
    i.e. the frame where each pedestrian's last observed position is the
    origin, which is what SocialDataset.__getitem__ provides via 'obs'.
 
    The evaluator converts to world coordinates using SocialDataset.reconstruct_abs
    before computing any metric.  This is the single reconstruction path for
    all models.
 
    Models that internally work in displacement space (Social-LSTM, Social-GAN,
    STGCNN) must cumsum their displacement outputs and return the result as
    normalised absolute positions.  The evaluator does not know or care how a
    model produces its output — it only requires this final shape and frame.
 
    Why normalised absolute and not raw world coordinates?
        Because 'obs' (the model's input) is in the normalised frame.  If a
        model outputs in the same frame it receives input in, no internal
        coordinate conversion is needed inside the model.  The single
        denormalisation step happens here, once, at evaluation time.
 
    Parameters
    ----------
    model  : nn.Module
        Must implement forward(obs, obs_rel, k) → FloatTensor [K, N, T, 2].
    config : dict
        Must contain an 'evaluation' sub-dict with optional key 'k_samples'
        (default 20).
    """
 
    def __init__(self, model, config: dict):
        self.model  = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.k = config.get('evaluation', {}).get('k_samples', 20)
 
    def evaluate(self, data_loader) -> dict:
        """
        Run evaluation over a full test DataLoader.
 
        Metrics are averaged over pedestrians (not over scenes or batches).
        Specifically, we accumulate the sum of per-pedestrian minimum errors
        and the total pedestrian count across the entire test set, then divide
        once at the end.  This matches the benchmark convention — reporting an
        average over all 1,536 ETH-UCY pedestrians, not an average of
        per-scene averages.
 
        Returns
        -------
        dict with keys 'ADE' and 'FDE' (float scalars).
        """
        self.model.eval()
 
        # Accumulate sum of per-pedestrian min errors and total ped count.
        # Dividing sum/count gives the correctly weighted mean regardless of
        # how many pedestrians appear in each scene or batch.
        ade_sum   = 0.0
        fde_sum   = 0.0
        ped_count = 0
 
        with torch.no_grad():
            for batch in data_loader:
                obs_list    = [o.to(self.device)    for o in batch['obs']]
                obs_rel_list = [r.to(self.device)   for r in batch['obs_rel']]
                target_list = [t.to(self.device)    for t in batch['pred']]
                origin_list = [o.to(self.device)    for o in batch['origin']]
 
                # Forward pass — one scene at a time (variable N_peds per scene).
                for obs, obs_rel, target, origin in zip(
                    obs_list, obs_rel_list, target_list, origin_list
                ):
                    n_peds = obs.shape[0]
 
                    # Model returns [K, N, pred_len, 2] in normalised frame.
                    preds_norm = self.model(obs, obs_rel, k=self.k)  # [K, N, T, 2]
 
                    # Convert to world coordinates — single reconstruction path.
                    # reconstruct_abs broadcasts origin [N, 1, 2] across K and T.
                    preds_abs = SocialDataset.reconstruct_abs(preds_norm, origin)
 
                    # Accumulate sum of per-pedestrian min errors.
                    # calculate_best_of_k returns the mean over N — multiply
                    # back by N to get the sum, which we accumulate across scenes.
                    ade_sum += calculate_best_of_k(preds_abs, target, 'ade').item() * n_peds
                    fde_sum += calculate_best_of_k(preds_abs, target, 'fde').item() * n_peds
                    ped_count += n_peds
 
        return {
            'ADE': ade_sum / ped_count,
            'FDE': fde_sum / ped_count,
        }
