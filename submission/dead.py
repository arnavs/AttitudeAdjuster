def keep_pair_to_discard_action(keep_idx_1, keep_idx_2):
    """Convert (keep_idx_1, keep_idx_2) into discard action index [0..9]."""
    pair = tuple(sorted([keep_idx_1, keep_idx_2]))
    return KEEP_PAIRS.index(pair)

def _apply_opp_model(self, probs, mask):
        if self.opp_postflop_actions < 20:
            return probs
        fold_rate = self.opp_postflop_folds / max(self.opp_postflop_actions, 1)
        if fold_rate > OPP_FOLD_THRESHOLD and mask[BET_SMALL] > 0 and mask[CHECK] > 0:
            shift = min(0.25, (fold_rate - OPP_FOLD_THRESHOLD) * 1.5)
            transfer = shift * probs[CHECK]
            probs[CHECK]     -= transfer
            probs[BET_SMALL] += transfer * 0.4
            probs[BET_MED]   += transfer * 0.35
            probs[BET_LARGE] += transfer * 0.25
        return probs