from minimax_h3_fl2v.frames import align_num_frames, duration_to_frames, frames_to_duration


def test_minimum_clip_is_124_frames():
    assert duration_to_frames(5.0) == 124
    assert align_num_frames(100) == 124
    assert abs(frames_to_duration(124) - 124 / 24) < 1e-9


def test_alignment_grid():
    for frames in (124, 141, 158, 175):
        assert (frames - 5) % 17 == 0
        assert align_num_frames(frames) == frames
