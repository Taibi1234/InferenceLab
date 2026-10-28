from scripts.benchmark import percentile


def test_percentile():
    values = [4.0, 1.0, 3.0, 2.0]

    assert percentile(values, 50) == 2.0
    assert percentile(values, 95) == 4.0