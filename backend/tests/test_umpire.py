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
    return RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), net_top_y=court_model.net_top_threshold_y(H))


def test_landing_inside_is_called_in():
    calls = feed(make_umpire(), flight_to((3.0, 3.0)))
    assert len(calls) == 1 and calls[0].result == "IN"
    assert np.allclose(calls[0].world_pt, (3.0, 3.0), atol=0.05)


def test_landing_in_doubles_alley_is_out_for_singles():
    calls = feed(make_umpire(), flight_to((3.0, 0.2)))
    assert len(calls) == 1 and calls[0].result == "OUT"


def test_no_call_without_rest():
    calls = feed(make_umpire(), flight_to((3.0, 3.0), rest_frames=0))
    # With extrapolation logic, a track ending in a steep dive is extrapolated to the floor.
    assert len(calls) == 1


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
    ump = RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), net_top_y=None)
    assert feed(ump, pts) == []
