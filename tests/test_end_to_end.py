"""Run the deployed loaders and both stage entry points with offline tiny weights."""

from tests.smoke_fd_llm import run_smoke


def test_real_cwru_to_19_class_diagnosis(tmp_path):
    summary = run_smoke(tmp_path)
    assert summary["input_shape"] == [2, 1024, 1]
    assert summary["class_score_shape"] == [2, 19]
    assert len(summary["alignment_history"]) == len(summary["diagnosis_history"]) == 1
    for file in ("alignment/best.pt", "diagnosis/best.pt", "evaluation/metrics.json",
                 "evaluation/confusion_matrix.png", "evaluation/tsne.png", "summary.json"):
        assert (tmp_path / file).is_file()
