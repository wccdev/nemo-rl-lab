
def test_extract_message_log_samples():
    from common.observability.validation_extract import extract_message_log_samples

    logs = [
        [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}],
        [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}],
    ]
    rewards = [1.0, 0.0]
    samples, dist, avg = extract_message_log_samples(logs, rewards, num_samples=2)
    assert len(samples) == 2
    assert samples[0]["user"] == "Q1"
    assert samples[0]["assistant"] == "A1"
    assert avg == 0.5
    assert dist == [{"reward": 0.0, "count": 1}, {"reward": 1.0, "count": 1}]


def test_validation_ctx_detects_step():
    from common.observability.validation_ctx import (
        active_validation_step,
        clear_validation_step,
        feed_log_text,
    )

    clear_validation_step()
    feed_log_text("noise\nStarting validation at step 42\n")
    assert active_validation_step() == 42
    clear_validation_step()
    assert active_validation_step() is None
