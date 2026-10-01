import numpy as np

from core import court_model
from core.umpire import RallyUmpire, judge, SINGLES, DOUBLES

CORNERS = [[322, 740], [999, 1029], [1008, 651], [1671, 735]]
H = court_model.homography_from_points(CORNERS)
H_INV = np.linalg.inv(H)


def test_singles_vs_doubles_sideline():
    assert judge((3.0, 0.3), SINGLES)[0] == "OUT"
    assert judge((3.0, 0.3), DOUBLES)[0] == "IN"


def test_line_is_in_and_close_call():
    result, _, close = judge((3.0, 0.46 - 0.01), SINGLES)    # on the singles sideline paint
    assert result == "IN" and close
    result, margin, close = judge((-0.05, 3.0), SINGLES)     # just past the baseline
    assert result == "OUT" and close and margin < 0
    result, margin, close = judge((3.0, 3.0), SINGLES)
    assert result == "IN" and not close and margin > 1.0


def _parabola_to_touchdown(land):
    """Single continuous parabola (rises above the net, falls under constant acceleration)
    down to the first frame at or past the landing height -- no synthetic slope change
    before it, so the only discontinuity in the flight is the impact itself."""
    pts, t = [], 0
    while True:
        y = land[1] - 400 + 0.9 * t * t - 13 * t
        x = land[0] - 150 + 5 * t
        if y >= land[1] and t > 7:
            break
        pts.append((x, y))
        t += 1
    return pts


def flight_to(world_landing, rest_frames=4):
    """Image-space flight: rises above the net threshold, falls onto the landing point in one
    continuous parabola, then a sudden full stop (the impact) for rest_frames frames."""
    land = court_model.world_to_img([world_landing], H)[0]
    pts = _parabola_to_touchdown(land) + [tuple(land)] * rest_frames
    return [(int(round(x)), int(round(y))) for x, y in pts]


def flight_bounce_to(world_landing, bounce_height=20.0, bounce_dx=40.0, n_bounce=15, rest_frames=4):
    """
    Touchdown (a real impact loses most of its energy, so the bounce arc starts much
    slower than the incoming fall), then a small secondary bounce that resettles a clear
    distance away. Returns (points, touchdown_px, final_rest_px); touchdown_px is the
    first sample after contact -- the earliest a discrete, ~60fps camera could ever place
    it, same convention as landing_point's own first-post-impact-sample report.
    """
    land = court_model.world_to_img([world_landing], H)[0]
    pts = _parabola_to_touchdown(land)
    contact = pts[-1]
    touchdown = None
    for k in range(1, n_bounce + 1):
        f = k / n_bounce
        p = (contact[0] + bounce_dx * f, contact[1] - bounce_height * 4 * f * (1 - f))
        pts.append(p)
        touchdown = touchdown or p
    rest_pt = pts[-1]
    pts += [rest_pt] * rest_frames
    return [(int(round(x)), int(round(y))) for x, y in pts], touchdown, rest_pt


def flight_slide_to(world_landing, slide_dx=40.0, n_slide=25, rest_frames=4):
    """
    Touchdown, then a decelerating horizontal skid (constant height, starting slow like a
    real friction-braked skid) to a stop some distance away. Returns (points, touchdown_px,
    final_rest_px); touchdown_px is the first sample after contact (see flight_bounce_to).
    """
    land = court_model.world_to_img([world_landing], H)[0]
    pts = _parabola_to_touchdown(land)
    contact = pts[-1]
    cur = contact
    touchdown = None
    for k in range(n_slide):
        remaining = n_slide - k
        dx = slide_dx * (2 * remaining - 1) / (n_slide * n_slide)  # decreasing steps, sums to slide_dx
        cur = (cur[0] + dx, contact[1])
        pts.append(cur)
        touchdown = touchdown or cur
    rest_pt = pts[-1]
    pts += [rest_pt] * rest_frames
    return [(int(round(x)), int(round(y))) for x, y in pts], touchdown, rest_pt


def feed(umpire, pts):
    calls = [umpire.update(i, p, True) for i, p in enumerate(pts)]
    calls.append(umpire.update(len(pts), None, False))   # track ends
    return [c for c in calls if c is not None]


def make_umpire():
    return RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), min_step_frames=1, net_top_y=court_model.net_top_threshold_y(H))


def test_landing_inside_is_called_in():
    calls = feed(make_umpire(), flight_to((3.0, 3.0)))
    assert len(calls) == 1 and calls[0].result == "IN"
    assert np.allclose(calls[0].world_pt, (3.0, 3.0), atol=0.05)


def test_landing_in_doubles_alley_is_out_for_singles():
    calls = feed(make_umpire(), flight_to((3.0, 0.2)))
    assert len(calls) == 1 and calls[0].result == "OUT"


def test_no_call_without_rest():
    # the track ends in a steep dive with no impact: nothing to judge by default
    assert feed(make_umpire(), flight_to((3.0, 3.0), rest_frames=0)) == []


def test_extrapolated_call_only_when_enabled_and_flagged_lost():
    ump = RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), min_step_frames=1, net_top_y=court_model.net_top_threshold_y(H),
                      allow_extrapolation=True)
    calls = feed(ump, flight_to((3.0, 3.0), rest_frames=0))
    assert len(calls) == 1 and calls[0].method == "lost"


def test_extrapolation_is_off_by_default():
    from core import config
    assert config.UMPIRE_ALLOW_EXTRAPOLATION is False
    assert make_umpire().allow_extrapolation is False


def test_no_call_when_resting_at_frame_border():
    pts = flight_to((3.0, 3.0))[:25]
    edge = [(1915, 600 + 10 * k) for k in range(8)] + [(1915, 670)] * 3
    assert feed(make_umpire(), pts + edge) == []


def test_low_flight_is_ignored():
    land = court_model.world_to_img([(3.0, 3.0)], H)[0]
    pts = [(int(land[0]) - 40 + 4 * k, int(land[1]) - 20 + 2 * k) for k in range(10)]
    pts += [(int(land[0]), int(land[1]))] * 4
    assert feed(make_umpire(), pts) == []


def test_no_call_when_resting_on_a_person():
    pts = flight_to((3.0, 3.0))
    land = pts[-1]
    box = (land[0] - 40, land[1] - 300, land[0] + 40, land[1] + 5)
    ump = make_umpire()
    calls = [ump.update(i, p, True, people_boxes=[box]) for i, p in enumerate(pts)]
    calls.append(ump.update(len(pts), None, False, people_boxes=[box]))
    assert [c for c in calls if c is not None] == []


def test_bounce_is_called_at_first_touchdown_not_final_rest():
    pts, touchdown, rest_pt = flight_bounce_to((3.0, 3.0))
    calls = feed(make_umpire(), pts)
    assert len(calls) == 1 and calls[0].result == "IN"
    call_px = calls[0].image_pt
    assert np.hypot(call_px[0] - touchdown[0], call_px[1] - touchdown[1]) < 10.0
    assert np.hypot(call_px[0] - rest_pt[0], call_px[1] - rest_pt[1]) > 20.0


def test_slide_is_called_at_first_touchdown_not_final_rest():
    pts, touchdown, rest_pt = flight_slide_to((3.0, 3.0))
    calls = feed(make_umpire(), pts)
    assert len(calls) == 1 and calls[0].result == "IN"
    call_px = calls[0].image_pt
    assert np.hypot(call_px[0] - touchdown[0], call_px[1] - touchdown[1]) < 10.0
    assert np.hypot(call_px[0] - rest_pt[0], call_px[1] - rest_pt[1]) > 20.0


def test_no_call_for_points_off_the_floor():
    # a resting blob high in the image (ceiling light) maps far outside the court
    pts = [(900, 60 + 30 * k) for k in range(8)] + [(900, 300)] * 4
    ump = RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), min_step_frames=1, net_top_y=None)
    assert feed(ump, pts) == []


# --- perspective-corrected rest speed, defaults, metadata ---------------------------------

def test_rest_speed_is_larger_near_the_camera():
    umpire = RallyUmpire(H_INV, rest_speed_px=2.0)
    near_cam, near_net = umpire._local_rest_speed(np.array([[900.0, 1000.0], [1000.0, 660.0]]))
    assert near_cam > 2.0 > near_net > 0
    assert near_net >= 1.5                     # never below the localisation-noise floor
    # without a homography the threshold is constant
    flat = RallyUmpire(None, rest_speed_px=2.0)._local_rest_speed(np.array([[900.0, 1000.0], [1000.0, 660.0]]))
    assert flat.tolist() == [2.0, 2.0]


def test_default_floor_region_covers_court_plus_buffer():
    from core import config
    x0, x1, y0, y1 = RallyUmpire(H_INV).floor_region
    b = config.UMPIRE_FLOOR_BUFFER_M
    assert (x0, y0) == (-b, -b)
    assert x1 == court_model.COURT_LENGTH + b and y1 == court_model.COURT_WIDTH + b


def test_landing_point_is_a_single_documented_method():
    assert "first ground contact" in RallyUmpire.landing_point.__doc__


def test_missing_metadata_falls_back_to_singles_with_warning(tmp_path, caplog):
    from core.umpire import match_type_from_metadata
    with caplog.at_level("WARNING", logger="core.umpire"):
        assert match_type_from_metadata("nope.mp4", metadata_path=tmp_path / "missing.csv") == SINGLES
    assert "assuming singles" in caplog.text


def test_metadata_reads_doubles(tmp_path):
    from core.umpire import match_type_from_metadata
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text("file,loai_tran\na.mp4,doi\nb.mp4,don\n", encoding="utf-8")
    assert match_type_from_metadata("x/a.mp4", metadata_path=csv_path) == DOUBLES
    assert match_type_from_metadata("x/b.mp4", metadata_path=csv_path) == SINGLES


# --- flight segmentation, call method, descent gate, uncertainty ---------------------------

def feed_flights(umpire, flights):
    """flights: list of point lists; each gets its own flight_id and the track never drops
    (a racket hit restarts the track inside one detect() call)."""
    calls, idx = [], 0
    for fid, pts in enumerate(flights, start=1):
        for p in pts:
            calls.append(umpire.update(idx, p, True, flight_id=fid))
            idx += 1
    calls.append(umpire.update(idx, None, False))
    return [c for c in calls if c is not None]


def test_flight_id_change_ends_the_previous_flight():
    # flight 1 lands; a hit-restart begins flight 2 without track_active ever dropping
    landing_flight = flight_to((3.0, 3.0))
    second = [(1200 + 5 * k, 300 + 4 * k) for k in range(6)]
    calls = feed_flights(make_umpire(), [landing_flight, second])
    assert len(calls) == 1 and calls[0].result == "IN"
    assert calls[0].frame_idx < len(landing_flight) + 1    # judged at the boundary, not at the end


def test_without_flight_id_hits_merge_flights():
    """Documents why flight_id exists: without it the second flight's points are appended to the first."""
    ump = make_umpire()
    pts = flight_to((3.0, 3.0)) + [(1200 + 5 * k, 300 + 4 * k) for k in range(6)]
    calls = [ump.update(i, p, True) for i, p in enumerate(pts)]
    assert all(c is None for c in calls)        # nothing ended yet: one long merged flight


def test_call_records_method_and_uncertainty():
    calls = feed(make_umpire(), flight_to((3.0, 3.0)))
    c = calls[0]
    assert c.method in ("contact", "contact_unconfirmed")
    assert c.uncertainty_m > 0
    d = c.to_dict()
    assert d["method"] == c.method and d["uncertainty_m"] == round(c.uncertainty_m, 3)


def test_confirmed_contact_vs_track_ending_right_after_impact():
    confirmed = feed(make_umpire(), flight_to((3.0, 3.0), rest_frames=4))[0]
    assert confirmed.method == "contact"
    pts = flight_to((3.0, 3.0), rest_frames=2)      # a single still step: the track ends before it can be confirmed
    unconfirmed = feed(make_umpire(), pts)
    assert len(unconfirmed) == 1 and unconfirmed[0].method == "contact_unconfirmed"


def test_min_descent_blocks_a_shallow_impact_and_lets_a_deep_one_through():
    pts = flight_to((3.0, 3.0))
    # the flight falls several hundred px from its apex, so any sane threshold passes...
    assert len(feed(RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), min_step_frames=1, min_descent_px=80,
                                net_top_y=court_model.net_top_threshold_y(H)), pts)) == 1
    # ...and an impossible one rejects the very same flight: the parameter is live
    assert feed(RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), min_step_frames=1, min_descent_px=5000,
                            net_top_y=court_model.net_top_threshold_y(H)), pts) == []


def test_far_half_close_call_uses_one_pixel_of_uncertainty():
    near_px = RallyUmpire(H_INV)._metres_per_px(np.array([court_model.world_to_img([(3.0, 3.0)], H)[0]]))[0]
    far_px = RallyUmpire(H_INV)._metres_per_px(np.array([court_model.world_to_img([(13.4, 3.0)], H)[0]]))[0]
    assert far_px > 0.10 > near_px           # the fixed 0.10 m threshold is under one far pixel
    # 0.12 m outside the far baseline: OUT, and "close" because one pixel there spans more than 0.12 m
    result, margin, close = judge((13.4 + 0.02 + 0.12, 3.0), SINGLES, half="far", uncertainty_m=far_px)
    assert result == "OUT" and close
    assert not judge((13.4 + 0.02 + 0.12, 3.0), SINGLES, half="far")[2]     # old behaviour


# --- a landing that the tracker sees as a hit: judged across the flight boundary -------------

def _boundary_case(after):
    """
    Old flight = the fall, whose last sample is on the floor; `after` = offsets (px) from that
    sample of the next flight's points (a hit or a bounce), which the tracker saw as a new flight.
    """
    land = court_model.world_to_img([(3.0, 3.0)], H)[0]
    fall = [(int(round(x)), int(round(y))) for x, y in _parabola_to_touchdown(land)] + [tuple(int(v) for v in land)]
    return fall, [(int(land[0]) + dx, int(land[1]) + dy) for dx, dy in after]


def test_bounce_at_the_flight_boundary_is_a_landing():
    fall, after = _boundary_case([(2, -2), (3, -1), (3, -1), (3, -1), (3, -1)])   # small hop, then rests
    calls = feed_flights(make_umpire(), [fall, after])
    assert len(calls) == 1 and calls[0].result == "IN"
    assert calls[0].method in ("contact", "contact_unconfirmed")
    assert np.hypot(calls[0].image_pt[0] - fall[-1][0], calls[0].image_pt[1] - fall[-1][1]) < 12


def test_racket_hit_at_the_flight_boundary_is_not_a_landing():
    fall, after = _boundary_case([(20, -60), (40, -140), (60, -240), (80, -350), (100, -470)])   # flies off fast
    assert feed_flights(make_umpire(), [fall, after]) == []


def test_boundary_waits_for_look_ahead_before_deciding():
    fall, after = _boundary_case([(2, -2), (3, -1), (3, -1), (3, -1), (3, -1)])
    ump = make_umpire()
    calls = []
    for i, p in enumerate(fall):
        calls.append(ump.update(i, p, True, flight_id=1))
    calls.append(ump.update(len(fall), after[0], True, flight_id=2))      # boundary: not decided yet
    assert all(c is None for c in calls)
    for j, p in enumerate(after[1:], start=1):
        calls.append(ump.update(len(fall) + j, p, True, flight_id=2))
    assert any(c is not None for c in calls)                              # decided once enough points followed


def test_boundary_is_resolved_when_the_track_ends_before_the_look_ahead_is_full():
    fall, after = _boundary_case([(2, -2)])
    calls = feed_flights(make_umpire(), [fall, after])
    assert len(calls) == 1 and calls[0].method == "contact_unconfirmed"


def test_landing_inside_the_new_flight_is_not_reported_twice():
    fall, after = _boundary_case([(2, -2), (3, -1), (3, -1), (3, -1), (3, -1), (3, -1)])
    calls = feed_flights(make_umpire(), [fall, after])
    assert len(calls) == 1


# --- shuttle seen falling, lost, then seen lying still on the floor ---------------------------

def _play(ump, segments):
    """segments: [(first_frame, [pts])]; each segment is one track that then ends (track_active drops)."""
    calls = []
    for first, pts in segments:
        for k, p in enumerate(pts):
            calls.append(ump.update(first + 5 * k, p, True))
        calls.append(ump.update(first + 5 * len(pts), None, False))
    return [c for c in calls if c is not None]


def _fall_and_rest(rest_dy=0, n_rest=4):
    land = court_model.world_to_img([(3.0, 3.0)], H)[0]
    fall = [(int(round(x)), int(round(y))) for x, y in _parabola_to_touchdown(land)][:-3]   # still falling
    rest = [(int(land[0]) + (k % 2), int(land[1]) + rest_dy) for k in range(n_rest)]
    return fall, rest


def test_fall_then_lying_still_is_called_at_the_first_resting_position():
    fall, rest = _fall_and_rest()
    calls = _play(make_umpire(), [(0, fall), (5 * len(fall) + 40, rest)])
    assert len(calls) == 1 and calls[0].method == "resting" and calls[0].result == "IN"
    assert calls[0].frame_idx == 5 * len(fall) + 40                 # the first resting detection
    assert calls[0].image_pt == tuple(rest[0])


def test_resting_needs_a_preceding_fall():
    _, rest = _fall_and_rest()
    assert _play(make_umpire(), [(1000, rest)]) == []


def test_resting_is_not_called_when_the_shuttle_reappears_too_late():
    fall, rest = _fall_and_rest()
    assert _play(make_umpire(), [(0, fall), (5 * len(fall) + 500, rest)]) == []


def test_resting_spot_far_above_where_the_fall_was_last_seen_is_not_a_landing():
    fall, rest = _fall_and_rest(rest_dy=-200)                          # rests 200 px higher than the fall ended
    assert _play(make_umpire(), [(0, fall), (5 * len(fall) + 40, rest)]) == []


def test_resting_needs_enough_still_points():
    fall, rest = _fall_and_rest(n_rest=2)
    assert _play(make_umpire(), [(0, fall), (5 * len(fall) + 40, rest)]) == []


def test_resting_can_be_turned_off():
    fall, rest = _fall_and_rest()
    ump = RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), min_step_frames=1, net_top_y=court_model.net_top_threshold_y(H),
                      allow_resting=False)
    assert _play(ump, [(0, fall), (5 * len(fall) + 40, rest)]) == []


def test_resting_is_not_called_twice_after_a_seen_landing():
    calls = _play(make_umpire(), [(0, flight_to((3.0, 3.0))), (200, _fall_and_rest()[1])])
    assert len(calls) == 1 and calls[0].method in ("contact", "contact_unconfirmed")


def test_resting_not_on_a_person():
    fall, rest = _fall_and_rest()
    ump = make_umpire()
    calls = [ump.update(5 * k, p, True) for k, p in enumerate(fall)] + [ump.update(5 * len(fall), None, False)]
    box = (rest[0][0] - 50, rest[0][1] - 300, rest[0][0] + 50, rest[0][1] + 10)
    first = 5 * len(fall) + 40
    calls += [ump.update(first + 5 * k, p, True, people_boxes=[box]) for k, p in enumerate(rest)]
    calls.append(ump.update(first + 5 * len(rest), None, False))
    assert [c for c in calls if c is not None] == []


# --- a detection on every frame is thinned to the spacing the landing logic was tuned for -------

def _sparse_and_dense(pts, step=5):
    """The same flight as detections `step` frames apart, and as one linearly interpolated point per frame."""
    sparse = [(step * k, p) for k, p in enumerate(pts)]
    dense = []
    for (f0, p0), (f1, p1) in zip(sparse[:-1], sparse[1:]):
        for f in range(f0, f1):
            w = (f - f0) / (f1 - f0)
            dense.append((f, (p0[0] + (p1[0] - p0[0]) * w, p0[1] + (p1[1] - p0[1]) * w)))
    dense.append(sparse[-1])
    return sparse, dense


def _play_frames(ump, frames):
    calls = [ump.update(f, p, True) for f, p in frames]
    calls.append(ump.update(frames[-1][0] + 1, None, False))
    return [c for c in calls if c is not None]


def test_dense_stream_gives_the_same_call_as_the_sparse_one():
    sparse, dense = _sparse_and_dense(flight_to((3.0, 3.0)))
    net = court_model.net_top_threshold_y(H)
    a = _play_frames(RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), net_top_y=net), sparse)
    b = _play_frames(RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), net_top_y=net), dense)
    assert len(a) == 1 and a == b


def test_decimate_is_a_no_op_for_sparse_points_and_keeps_the_first_of_each_step():
    from core.umpire import FlightPoint
    ump = make_umpire()
    ump.min_step_frames = 5
    sparse = [FlightPoint(5 * k, (k, k), False) for k in range(6)]
    assert ump._decimate(sparse) is sparse or ump._decimate(sparse) == sparse
    dense = [FlightPoint(f, (f, f), False) for f in range(0, 23)]
    assert [p.frame_idx for p in ump._decimate(dense)] == [0, 5, 10, 15, 20]
    uneven = [FlightPoint(f, (f, f), False) for f in (0, 1, 7, 8, 9, 14)]
    assert [p.frame_idx for p in ump._decimate(uneven)] == [0, 7, 14]


def test_dense_boundary_is_resolved_after_enough_thinned_look_ahead_points():
    fall, after = _boundary_case([(2, -2)] + [(3, -1)] * 40)         # the shuttle bounces, then rests
    net = court_model.net_top_threshold_y(H)
    ump = RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), net_top_y=net)
    sparse_fall, dense_fall = _sparse_and_dense(fall)
    last_fall_frame = dense_fall[-1][0]
    calls = [ump.update(f, p, True, flight_id=1) for f, p in dense_fall]
    dense_after = [(last_fall_frame + 1 + k, p) for k, p in enumerate(after)]       # one point per frame
    calls += [ump.update(f, p, True, flight_id=2) for f, p in dense_after[:4]]
    assert all(c is None for c in calls)        # 4 frames of look-ahead are not yet rest_frames + 1 spaced points
    calls += [ump.update(f, p, True, flight_id=2) for f, p in dense_after[4:20]]
    calls.append(ump.update(dense_after[-1][0] + 1, None, False))
    made = [c for c in calls if c is not None]
    assert len(made) == 1 and made[0].result == "IN"
