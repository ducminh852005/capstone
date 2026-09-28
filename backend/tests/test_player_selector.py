from core.player_tracker import PlayerSelector

FPS = 10  # small numbers: window 100 frames, a track needs 5 in-court frames to be selected


def run(sel, frames):
    return [sel.update(i, obs) for i, obs in enumerate(frames)]


def test_selects_single_player_and_ignores_audience_and_passerby():
    sel = PlayerSelector(fps=FPS)
    frames = []
    for i in range(60):
        obs = {1: ((3.0, 3.0), True), 2: ((3.0, 8.0), False)}
        if 20 <= i < 26:
            obs[3] = ((1.0, 1.0), True)
        frames.append(obs)
    out = run(sel, frames)
    assert out[2] == {}                        # not confirmed yet
    assert all(o == {1: 1} for o in out[5:])   # one stable player id on the real player's track


def test_reid_after_short_track_loss():
    sel = PlayerSelector(fps=FPS)
    frames = [{1: ((3.0, 3.0), True)} for _ in range(30)]
    frames += [{} for _ in range(10)]                        # lost for 1 s
    frames += [{7: ((3.5, 3.2), True)} for _ in range(5)]    # new ByteTrack id nearby
    out = run(sel, frames)
    assert out[29] == {1: 1}
    assert out[40] == {1: 7}                                 # same player id, new track


def test_far_new_track_is_not_merged_and_slot_is_released():
    sel = PlayerSelector(fps=FPS, reid_window_s=2.0)
    frames = [{1: ((3.0, 3.0), True)} for _ in range(30)]
    frames += [{9: ((0.2, 6.0), True)} for _ in range(60)]   # someone else, far from the last position
    out = run(sel, frames)
    assert out[31] == {}                                     # not merged into player 1
    assert out[-1] == {2: 9}                                 # after the re-id window a new player id is issued
