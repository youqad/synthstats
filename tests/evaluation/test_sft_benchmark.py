import pytest

from synthstats.data.sft_loader import SFTExample
from synthstats.evaluation.sft_benchmark import (
    BenchmarkResult,
    evaluate_on_sft,
    extract_program,
    normalize_code,
)


class TestNormalizeCode:
    def test_strips_whitespace(self):
        code = "   \n  x = 1  \n   "
        assert normalize_code(code) == "x = 1"

    def test_removes_blank_lines(self):
        code = "x = 1\n\n\ny = 2"
        result = normalize_code(code)
        assert "\n\n" not in result
        assert "x = 1" in result
        assert "y = 2" in result

    def test_preserves_comments(self):
        code = "x = 1  # this is a comment\ny = 2  # another"
        result = normalize_code(code)
        assert "x = 1" in result
        assert "y = 2" in result

    def test_preserves_code_structure(self):
        code = """def foo(x):
    return x + 1"""
        result = normalize_code(code)
        assert "def foo(x):" in result
        assert "return x + 1" in result


class TestExtractProgram:
    def test_extracts_python_code_block(self):
        text = "Here's code:\n```python\nx = 1\n```"
        program = extract_program(text)
        assert program == "x = 1"

    def test_extracts_submit_program_block(self):
        text = "<submit_program>\nx = 1\n</submit_program>"
        program = extract_program(text)
        assert program == "x = 1"

    def test_prefers_submit_program_block_when_both_present(self):
        text = """<submit_program>
x = 1
</submit_program>

```python
y = 2
```"""
        program = extract_program(text)
        assert program == "x = 1"

    def test_returns_none_without_code_block(self):
        text = "No code here"
        assert extract_program(text) is None

    def test_handles_multiline_code(self):
        text = """```python
def foo():
    x = 1
    return x
```"""
        program = extract_program(text)
        assert "def foo():" in program
        assert "return x" in program


class TestBenchmarkResult:
    def test_to_dict(self):
        result = BenchmarkResult(
            num_examples=10,
            exact_match=0.1,
            program_match=0.3,
            mean_reward=0.5,
            reward_std=0.2,
            generation_success=0.8,
        )

        d = result.to_dict()

        assert d["eval/num_examples"] == 10.0
        assert d["eval/exact_match"] == 0.1
        assert d["eval/program_match"] == 0.3
        assert d["eval/mean_reward"] == 0.5
        assert d["eval/reward_std"] == 0.2
        assert d["eval/generation_success"] == 0.8


class TestEvaluateOnSFT:
    @pytest.fixture
    def mock_policy(self):
        class MockPolicy:
            def __init__(self, completions: list[str]):
                self.completions = completions
                self._idx = 0

            def __call__(self, prompt: str, temperature: float = 0.7):
                if self._idx < len(self.completions):
                    completion = self.completions[self._idx]
                    self._idx += 1
                else:
                    completion = "no output"

                action = {"type": "submit_program", "payload": completion}
                return action, -0.5, 0.1  # action, logp, entropy

        return MockPolicy

    @pytest.fixture
    def sample_examples(self):
        return [
            SFTExample(
                prompt="Question 1",
                completion="```python\nx = 1\n```",
                thinking=None,
                program="x = 1",
            ),
            SFTExample(
                prompt="Question 2",
                completion="```python\ny = 2\n```",
                thinking=None,
                program="y = 2",
            ),
        ]

    def test_basic_evaluation(self, mock_policy, sample_examples):
        policy = mock_policy(
            [
                "```python\nx = 1\n```",
                "```python\ny = 2\n```",
            ]
        )

        result = evaluate_on_sft(
            policy,
            sample_examples,
            judge=None,
            show_progress=False,
        )

        assert result.num_examples == 2
        assert result.generation_success == 1.0
        assert result.program_match == 1.0

    def test_no_match_evaluation(self, mock_policy, sample_examples):
        policy = mock_policy(
            [
                "```python\nwrong = 'code'\n```",
                "```python\nalso_wrong = True\n```",
            ]
        )

        result = evaluate_on_sft(
            policy,
            sample_examples,
            judge=None,
            show_progress=False,
        )

        assert result.num_examples == 2
        assert result.generation_success == 1.0
        assert result.program_match == 0.0

    def test_partial_generation_success(self, mock_policy, sample_examples):
        policy = mock_policy(
            [
                "```python\nx = 1\n```",
                "no code block here",  # no valid program
            ]
        )

        result = evaluate_on_sft(
            policy,
            sample_examples,
            judge=None,
            show_progress=False,
        )

        assert result.generation_success == 0.5

    def test_submit_program_counts_as_success(self, mock_policy, sample_examples):
        policy = mock_policy(
            [
                "<submit_program>\nx = 1\n</submit_program>",
                "<submit_program>\ny = 2\n</submit_program>",
            ]
        )

        result = evaluate_on_sft(
            policy,
            sample_examples,
            judge=None,
            show_progress=False,
        )

        assert result.generation_success == 1.0
        assert result.program_match == 1.0

    def test_structured_submit_is_program_generation(self, mock_policy, sample_examples):
        policy = mock_policy(
            [
                "x = 1",
                "y = 2",
            ]
        )

        result = evaluate_on_sft(
            policy,
            sample_examples,
            judge=None,
            show_progress=False,
        )

        assert result.generation_success == 1.0
        assert result.program_match == 1.0

    def test_structured_submit_rejects_trivial_id(self, mock_policy, sample_examples):
        policy = mock_policy(
            [
                "done",
                "done",
            ]
        )

        result = evaluate_on_sft(
            policy,
            sample_examples,
            judge=None,
            show_progress=False,
        )

        assert result.generation_success == 0.0
        assert result.program_match == 0.0

    def test_max_examples_limit(self, mock_policy, sample_examples):
        policy = mock_policy(
            [
                "```python\nx = 1\n```",
            ]
        )

        result = evaluate_on_sft(
            policy,
            sample_examples,
            judge=None,
            max_examples=1,
            show_progress=False,
        )

        assert result.num_examples == 1

    def test_per_example_storage(self, mock_policy, sample_examples):
        policy = mock_policy(
            [
                "```python\nx = 1\n```",
                "```python\ny = 2\n```",
            ]
        )

        result = evaluate_on_sft(
            policy,
            sample_examples,
            judge=None,
            store_per_example=True,
            show_progress=False,
        )

        assert result.per_example is not None
        assert len(result.per_example) == 2
        assert "generated_program" in result.per_example[0]
        assert "is_program_match" in result.per_example[0]

    def test_empty_examples(self, mock_policy):
        policy = mock_policy([])

        result = evaluate_on_sft(
            policy,
            [],
            judge=None,
            show_progress=False,
        )

        assert result.num_examples == 0
        assert result.exact_match == 0.0
        assert result.program_match == 0.0
        assert result.generation_success == 0.0
        assert result.mean_reward == 0.0

    def test_generation_with_empty_reference(self, mock_policy):
        examples = [
            SFTExample(
                prompt="Question 1",
                completion="```python\nx = 1\n```",
                thinking=None,
                program="",
            ),
        ]
        policy = mock_policy(["```python\nx = 1\n```"])

        result = evaluate_on_sft(
            policy,
            examples,
            judge=None,
            show_progress=False,
        )

        assert result.num_examples == 1
        assert result.generation_success == 1.0
        assert result.program_match == 0.0

    def test_whitespace_normalization_in_match(self, mock_policy, sample_examples):
        policy = mock_policy(
            [
                "```python\n  x = 1  \n```",
                "```python\n\ny = 2\n\n```",
            ]
        )

        result = evaluate_on_sft(
            policy,
            sample_examples,
            judge=None,
            show_progress=False,
        )

        assert result.program_match == 1.0


class TestIntegrationWithJudge:
    @pytest.fixture
    def mock_judge(self):
        class MockJudge:
            def score(self, task_name, trajectory, artifacts):
                program = artifacts.get("program", "")
                reward = len(program) / 100.0
                return type("Reward", (), {"total": reward})()

        return MockJudge()

    def test_evaluation_with_judge(self, mock_judge):
        examples = [
            SFTExample(
                prompt="Q",
                completion="```python\nx = 1\n```",
                thinking=None,
                program="x = 1",
            ),
        ]

        class Policy:
            def __call__(self, prompt, temperature=0.7):
                return {"payload": "```python\nx = 1\n```"}, -0.5, 0.1

        result = evaluate_on_sft(
            Policy(),
            examples,
            judge=mock_judge,
            show_progress=False,
        )

        assert result.mean_reward > 0

    def test_judge_scores_structured_submit_no_fence(self, mock_judge):
        example = SFTExample(
            prompt="Q",
            completion="```python\nx = 1\n```",
            thinking=None,
            program="x = 1",
        )

        class Policy:
            def __call__(self, prompt, temperature=0.7):
                return {"type": "submit_program", "payload": "x = 1"}, -0.5, 0.1

        result = evaluate_on_sft(
            Policy(),
            [example],
            judge=mock_judge,
            show_progress=False,
        )

        assert result.generation_success == 1.0
        assert result.program_match == 1.0
        assert result.mean_reward > 0

    def test_best_of_n_keeps_highest_reward_program(self):
        example = SFTExample(
            prompt="Q",
            completion="```python\nx = 1\n```",
            thinking=None,
            program="x = 1",
        )

        class Policy:
            def __init__(self):
                self.calls = 0

            def __call__(self, prompt, temperature=0.7):
                self.calls += 1
                if self.calls == 1:
                    return {"payload": "```python\nx = 1\n```"}, -0.5, 0.1
                return {"payload": "```python\nx = 100\n```"}, -0.5, 0.1

        class Judge:
            def score(self, task_name, trajectory, artifacts):
                program = artifacts["program"]
                reward = 0.1 if program == "x = 1" else 0.9
                return type("Reward", (), {"total": reward})()

        result = evaluate_on_sft(
            Policy(),
            [example],
            judge=Judge(),
            num_samples=2,
            store_per_example=True,
            show_progress=False,
        )

        assert result.mean_reward == pytest.approx(0.9)
        assert result.per_example is not None
        assert result.per_example[0]["generated_program"] == "x = 100"

    def test_judge_float_reward_is_supported(self):
        example = SFTExample(
            prompt="Q",
            completion="```python\nx = 1\n```",
            thinking=None,
            program="x = 1",
        )

        class Policy:
            def __call__(self, prompt, temperature=0.7):
                return {"payload": "```python\nx = 1\n```"}, -0.5, 0.1

        class FloatJudge:
            def score(self, task_name, trajectory, artifacts):
                return 1.23

        result = evaluate_on_sft(
            Policy(),
            [example],
            judge=FloatJudge(),
            show_progress=False,
        )

        assert result.mean_reward == pytest.approx(1.23)

    def test_generation_exception_does_not_abort_example(self):
        example = SFTExample(
            prompt="Q",
            completion="```python\nx = 1\n```",
            thinking=None,
            program="x = 1",
        )

        class FlakyPolicy:
            def __init__(self):
                self.calls = 0

            def __call__(self, prompt, temperature=0.7):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transient generation error")
                return {"payload": "```python\nx = 1\n```"}, -0.5, 0.1

        result = evaluate_on_sft(
            FlakyPolicy(),
            [example],
            judge=None,
            num_samples=2,
            show_progress=False,
        )

        assert result.generation_success == 1.0
        assert result.program_match == 1.0
