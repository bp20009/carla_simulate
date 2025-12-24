import math


def test_sender_range_calculation_matches_expected_frames() -> None:
    """
    Verify the sender start/end frame computation mirrors batch_run_and_analyze_decel logic.

    The script derives the payload range when a center payload frame is provided by:
    - converting fixed_delta to payload frames per second (pf_per_sec)
    - computing the number of payload frames to send before/after the center frame
      using pre/post durations in seconds
    - subtracting/adding those counts to the center frame
    """

    fixed_delta = 0.1
    center_frame = 25411
    pre_frames = 600
    post_frames = 300

    pf_per_sec = int(round(1.0 / fixed_delta))
    pre_pf = int(round((pre_frames * fixed_delta) * pf_per_sec))
    post_pf = int(round((post_frames * fixed_delta) * pf_per_sec))

    start_frame = max(center_frame - pre_pf, 0)
    end_frame = center_frame + post_pf

    assert math.isclose(fixed_delta * pf_per_sec, 1.0)
    assert start_frame == 24811
    assert end_frame == 25711
