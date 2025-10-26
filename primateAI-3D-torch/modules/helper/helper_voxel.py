def rep_3d(v, nReps):
    v_rep = v[:, None, :].expand(-1, nReps, 3).reshape(-1, 3)

    return v_rep

def rep_1d(v, nReps):
    v_rep = v[..., None].expand(-1, nReps).reshape(-1)

    return v_rep

