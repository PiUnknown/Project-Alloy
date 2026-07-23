from alloy.labeling.weak_labels import combined_score, weak_label


def test_combined_score_sums_question_and_answer():
    assert combined_score(question_score=3, answer_score=5) == 8


def test_weak_label_assigns_correct_quantile_buckets():
    scores = list(range(100))  # 0..99, evenly spaced
    labels = weak_label(scores, top_quantile=0.75, bottom_quantile=0.25)
    assert labels[99] == 1   # highest score -> positive
    assert labels[0] == 0    # lowest score -> negative
    assert labels[50] == -1  # middle -> ambiguous, excluded


def test_weak_label_never_returns_out_of_range_labels():
    scores = [1, 5, 2, 9, 0, 7, 3]
    labels = weak_label(scores, top_quantile=0.7, bottom_quantile=0.3)
    assert set(labels).issubset({-1, 0, 1})