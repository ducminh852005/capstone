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


def flight_to(world_landing, rest_frames=4):
    """Image-space flight: rises above the net threshold, falls onto the landing point, then rests."""
    land = court_model.world_to_img([world_landing], H)[0]
    pts = [(land[0] - 150 + 5 * t, land[1] - 400 + 0.9 * t * t - 13 * t) for t in range(30)]
    fall_end = pts[-1]
    for k in range(1, 6):
        pts.append((fall_end[0] + (land[0] - fall_end[0]) * k / 5, fall_end[1] + (land[1] - fall_end[1]) * k / 5))
    pts += [tuple(land)] * rest_frames
    return [(int(round(x)), int(round(y))) for x, y in pts]


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
    assert feed(make_umpire(), flight_to((3.0, 3.0), rest_frames=0)) == []   # e.g. hit again


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


def test_no_call_for_points_off_the_floor():
    # a resting blob high in the image (ceiling light) maps far outside the court
    pts = [(900, 60 + 30 * k) for k in range(8)] + [(900, 300)] * 4
    ump = RallyUmpire(H_INV, SINGLES, frame_size=(1080, 1920), net_top_y=None)
    assert feed(ump, pts) == []
