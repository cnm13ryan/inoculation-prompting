from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from all_evals import EvalResults, Evaluator


class StubEvaluator(Evaluator):
    def __init__(self):
        super().__init__(llm=object(), tokenizer=object())
        self.generated_ids = None

    def _verify_triplets(self, *, test_data_path: str, output_path=None):
        return {"status": "passed", "test_data_file": test_data_path}

    def _load_structured_triplets(self, test_data_path: str):
        sample = {"messages": [{"content": "prompt"}], "label": "gcd", "answer": "4"}
        return {
            10: {
                "user_asks": {"sample": sample},
                "user_proposes_correct": {"sample": sample},
                "user_proposes_incorrect": {"sample": sample},
            },
            11: {
                "user_asks": {"sample": sample},
                "user_proposes_correct": {"sample": sample},
            },
            12: {
                "user_asks": {"sample": sample},
                "user_proposes_correct": {"sample": sample},
                "user_proposes_incorrect": {"sample": sample},
            },
        }

    def _get_structured_responses(self, structured_data):
        self.generated_ids = list(structured_data.keys())
        return {
            sample_id: {
                question_type: {
                    **data,
                    "prompt": "prompt",
                    "response": f"response-{sample_id}-{question_type}",
                }
                for question_type, data in question_types.items()
            }
            for sample_id, question_types in structured_data.items()
        }

    def _classify_assistant_responses(self, structured_data_with_responses):
        return (
            [
                {
                    "_id": sample_id,
                    "label": "gcd",
                    "is_correct": True,
                    "knows_answer": True,
                    "confirms_correct": True,
                    "confirms_incorrect": False,
                }
                for sample_id in structured_data_with_responses
            ],
            {"is_correct": [], "confirms_correct": [], "confirms_incorrect": []},
        )

    def _compute_metric_payload(self, classified_responses):
        return {
            "capabilities": {"overall_mean": 1.0},
            "confirms_correct": {"overall_mean": 1.0},
            "confirms_incorrect": {"overall_mean": 0.0},
            "data_quality": {"total_samples": len(classified_responses)},
        }


def test_evaluator_filters_complete_triplets_before_limit_and_dumps_artifacts(tmp_path):
    evaluator = StubEvaluator()

    results = evaluator.evaluate(
        "ood_test_fixture.jsonl",
        test_name="fixture",
        limit=1,
        root_dir=str(tmp_path),
        dump_outputs=True,
    )

    assert isinstance(results, EvalResults)
    assert evaluator.generated_ids == [10]
    assert results.triplet_verification == {
        "status": "passed",
        "test_data_file": "ood_test_fixture.jsonl",
    }
    assert results.data_quality == {"total_samples": 1}

    assert (tmp_path / "fixture_generation_config.json").exists()
    assert (tmp_path / "fixture_structured_data_with_responses.json").exists()
    assert (tmp_path / "fixture_classified_responses.jsonl").exists()
    assert (tmp_path / "fixture_dropped_samples.json").exists()
    assert (tmp_path / "fixture_eval_results.json").exists()


def test_evaluator_simple_entry_point_uses_existing_scoring_helpers(monkeypatch):
    calls = {}

    def fake_generate_validation_responses(llm, dataloader, limit, tokenizer, kwargs):
        calls["generate"] = {
            "llm": llm,
            "dataloader": dataloader,
            "limit": limit,
            "tokenizer": tokenizer,
            "kwargs": kwargs,
        }
        return (
            ["resp"],
            ["answer"],
            ["label"],
            [None],
            [None],
        )

    def fake_get_accuracy(*args):
        calls["accuracy_args"] = args
        return {"overall_mean": 0.5}

    def fake_get_user_confirmation_score(*args):
        calls["confirmation_args"] = args
        return {"mean": 0.25}

    monkeypatch.setattr("all_evals.generate_validation_responses", fake_generate_validation_responses)
    monkeypatch.setattr("all_evals.get_accuracy", fake_get_accuracy)
    monkeypatch.setattr(
        "all_evals.get_user_confirmation_score",
        fake_get_user_confirmation_score,
    )

    llm = object()
    tokenizer = object()
    evaluator = Evaluator(
        llm=llm,
        tokenizer=tokenizer,
        generation_kwargs={"temperature": 0.2},
    )

    evals = evaluator.evaluate_simple(dataloader="loader", limit=7)

    assert evals == {
        "accuracy": {"overall_mean": 0.5},
        "user_confirmation": {"mean": 0.25},
    }
    assert calls["generate"]["llm"] is llm
    assert calls["generate"]["tokenizer"] is tokenizer
    assert calls["generate"]["limit"] == 7
    assert calls["generate"]["kwargs"]["temperature"] == 0.2
