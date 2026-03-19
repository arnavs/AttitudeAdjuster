def keep_pair_to_discard_action(keep_idx_1, keep_idx_2):
    """Convert (keep_idx_1, keep_idx_2) into discard action index [0..9]."""
    pair = tuple(sorted([keep_idx_1, keep_idx_2]))
    return KEEP_PAIRS.index(pair)