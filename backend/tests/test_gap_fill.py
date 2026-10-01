import numpy as np

from core import court_model
from core.gap_fill import Detection, court_constraint, fill_gaps, split_chains
from core.shuttle_tracker import ShuttleDetector


def truth(t):
    """Ballistic flight in image px (gravity pulls y down)."""
    return 400.0 + 12.0 * t, 700.0 - 25.0 * t + 0.5 * t * t


def dets_at(frames, flight_id=1, start_first=None):
    out = []
    for i, t in enumerate(frames):
        out.append(Detection(t, truth(t), flight_id, start_first if i == 0 else None))
    return out


def err(fp):
    tx, ty = truth(fp.frame_idx)
    return float(np.hypot(fp.pt[0] - tx, fp.pt[1] - ty))


def test_stride_gaps_inside_a_chain_follow_the_parabola():
    filled = fill_gaps(dets_at([0, 8, 16, 24, 32]), max_link_frames=12)
    assert [p.frame_idx for p in filled] == [f for f in range(1, 32) if f % 8]
    assert all(p.kind == "stride" for p in filled)
    assert max(err(p) for p in filled) < 1e-6


def test_bridge_over_a_missed_detection():
    dets = dets_at([0, 8, 16, 40, 48, 56])          # detections at 24 and 32 are missing
    filled = fill_gaps(dets, max_link_frames=12)
    bridge = [p for p in filled if p.kind == "bridge"]
    assert [p.frame_idx for p in bridge] == list(range(17, 40))
    assert max(err(p) for p in bridge) < 1e-6


def test_bridge_needs_two_detections_on_each_side():
    only_one_before = dets_at([16, 40, 48, 56])
    assert not [p for p in fill_gaps(only_one_before, max_link_frames=12) if p.kind == "bridge"]
    only_one_after = dets_at([0, 8, 16, 40])
    assert not [p for p in fill_gaps(only_one_after, max_link_frames=12) if p.kind == "bridge"]


def test_bridge_refused_when_the_gap_is_too_long():
    dets = dets_at([0, 8, 16, 96, 104, 112])
    assert not [p for p in fill_gaps(dets, max_link_frames=12, max_gap_frames=48) if p.kind == "bridge"]
    assert [p for p in fill_gaps(dets, max_link_frames=12, max_gap_frames=100) if p.kind == "bridge"]


def test_bridge_refused_when_a_hit_changed_the_flight_inside_the_gap():
    before = dets_at([0, 8, 16])
    # after the gap the shuttle flies back the way it came, on a different parabola
    after = [Detection(40 + 8 * k, (truth(16)[0] - 12.0 * (8 * k + 24), 500.0 + 6.0 * k), 2) for k in range(3)]
    filled = fill_gaps(before + after, max_link_frames=12)
    assert not [p for p in filled if p.kind == "bridge"]


def test_bridge_refused_when_the_next_chain_starts_at_a_hit():
    dets = dets_at([0, 8, 16]) + [Detection(t, truth(t), 2, "physics" if t == 40 else None) for t in (40, 48, 56)]
    assert not [p for p in fill_gaps(dets, max_link_frames=12) if p.kind == "bridge"]
    dets_new = dets_at([0, 8, 16]) + [Detection(t, truth(t), 2, "new" if t == 40 else None) for t in (40, 48, 56)]
    assert [p for p in fill_gaps(dets_new, max_link_frames=12) if p.kind == "bridge"]


def test_chains_split_at_a_flight_change_even_when_close():
    dets = dets_at([0, 8, 16]) + dets_at([24, 32], flight_id=2)
    chains = split_chains(dets, max_link_frames=12)
    assert [len(c) for c in chains] == [3, 2]


def test_unknown_flight_id_does_not_split_a_chain():
    dets = [Detection(0, truth(0), 1), Detection(8, truth(8), None), Detection(16, truth(16), 1)]
    assert len(split_chains(dets, max_link_frames=12)) == 1


def test_court_constraint_rejects_the_whole_gap():
    dets = dets_at([0, 8, 16, 24])
    roi = (0, 0, 2000, 560)          # the first gap (frames 1..7) starts at y = 675, outside the roi
    filled = fill_gaps(dets, allowed=court_constraint(roi=roi), max_link_frames=12)
    # a gap with any point outside the roi is left empty as a whole; the others are kept
    assert filled and all(p.frame_idx > 8 for p in filled)
    for p in filled:
        assert p.pt[1] <= 560
    unconstrained = fill_gaps(dets, max_link_frames=12)
    assert len(filled) < len(unconstrained)


def test_court_constraint_rules_out_points_below_the_floor_but_not_high_ones():
    H = court_model.homography_from_points([[322, 740], [999, 1029], [1008, 651], [1671, 735]])
    H_inv = np.linalg.inv(H)
    allowed = court_constraint(H_inv=H_inv)
    centre = np.array(court_model.world_to_img([[court_model.COURT_LENGTH / 2, 3.05]], H)[0], float)
    near_mid = np.array(court_model.world_to_img([[-3.0, 3.05]], H)[0], float)
    away = (centre - near_mid) / np.linalg.norm(centre - near_mid)      # image direction from the near edge into the court
    assert allowed(tuple(centre))
    # a shuttle high above the far end of the court projects onto the floor far behind it: fine
    assert allowed(tuple(centre + 300.0 * away))
    # just outside the near edge (on the far side from the court): under the floor
    assert not allowed(tuple(near_mid - 40.0 * away))
    assert court_constraint()((5.0, 5.0))       # no roi, no homography: everything is allowed


def test_detector_fill_gaps_uses_its_trajectory_and_meta():
    det = ShuttleDetector(backend="cv")
    frames = [0, 8, 16, 40, 48, 56]
    det.trajectory = [None] * 57
    det.trajectory_meta = [(None, None)] * 57
    for i, t in enumerate(frames):
        x, y = truth(t)
        det.trajectory[t] = (int(round(x)), int(round(y)))
        det.trajectory_meta[t] = (1, "new" if i == 0 else None)
    filled = det.fill_gaps()          # cv backend: stride 1, link = 2 frames: every detection is its own chain
    assert filled == []               # single-detection chains cannot be bridged
    det._source.batch_stride = 8      # as for tracknet
    filled = det.fill_gaps()
    assert len(filled) > 0 and {p.kind for p in filled} == {"stride", "bridge"}
    assert max(err(p) for p in filled) < 1.5       # detections were rounded to whole pixels
    # tail window: only recent calls are looked at
    assert all(p.frame_idx > 30 for p in det.fill_gaps(tail=26))


def test_fit_cache_returns_exactly_what_a_fresh_fit_would_and_is_hit_on_repeats():
    from core import gap_fill
    window = dets_at([0, 8, 16, 24])
    gap_fill._fit_cached.cache_clear()
    first = gap_fill._fit(window)
    uncached = gap_fill._fit_cached.__wrapped__(tuple((d.frame_idx, d.pt[0], d.pt[1]) for d in window))
    for a, b in zip(first, uncached):
        assert np.array_equal(a, b)
    hits_before = gap_fill._fit_cached.cache_info().hits
    gap_fill._fit(window)
    assert gap_fill._fit_cached.cache_info().hits == hits_before + 1


def test_repeated_fill_after_one_new_detection_only_fits_the_new_gaps():
    from core import gap_fill
    base = dets_at([0, 8, 16, 24, 32, 40])
    gap_fill._fit_cached.cache_clear()
    first = fill_gaps(base, max_link_frames=12)
    misses = gap_fill._fit_cached.cache_info().misses
    grown = base + dets_at([48])
    second = fill_gaps(grown, max_link_frames=12)
    new_misses = gap_fill._fit_cached.cache_info().misses - misses
    assert new_misses <= 2                               # the last gaps' windows changed, the rest are cached
    # gaps whose fit window did not change are bit-identical; the newest gap is re-fitted with the new point
    assert [p for p in second if p.frame_idx <= 32] == [p for p in first if p.frame_idx <= 32]
    later = [p for p in second if 32 < p.frame_idx <= 40], [p for p in first if 32 < p.frame_idx <= 40]
    assert all(abs(a.pt[0] - b.pt[0]) < 1e-6 and abs(a.pt[1] - b.pt[1]) < 1e-6 for a, b in zip(*later))
