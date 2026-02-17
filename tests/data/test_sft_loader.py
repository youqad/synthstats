import json
from pathlib import Path

import pytest

from synthstats.data.sft_loader import (
    SFTExample,
    load_sft_jsonl,
    parse_completion,
    sft_to_buffer_entry,
)


class TestParseCompletion:
    def test_full_completion_with_think_and_code(self):
        completion = """<think>
This is my reasoning about the problem.
I think we should use a Poisson model.
</think>

Here's the model:

```python
import pymc as pm

def model(data):
    with pm.Model() as m:
        x = pm.Normal('x', 0, 1)
        return m
```
"""
        thinking, program = parse_completion(completion)

        assert thinking is not None
        assert "Poisson model" in thinking
        assert "import pymc" in program
        assert "pm.Normal" in program

    def test_completion_without_think_tags(self):
        completion = """```python
import pymc as pm
x = 1
```"""
        thinking, program = parse_completion(completion)

        assert thinking is None
        assert "import pymc" in program

    def test_completion_without_code_block(self):
        completion = """<think>
Some thoughts here.
</think>

No code block in this one.
"""
        thinking, program = parse_completion(completion)

        assert thinking is not None
        assert "Some thoughts" in thinking
        assert program == ""

    def test_submit_program_takes_precedence_over_code_fence(self):
        completion = """<think>Reasoning</think>

<submit_program>
import pymc as pm
x = 1
</submit_program>

```python
should_not_be_chosen = True
```"""
        thinking, program = parse_completion(completion)

        assert thinking is not None
        assert "Reasoning" in thinking
        assert "import pymc" in program
        assert "should_not_be_chosen" not in program

    def test_submit_program_without_think(self):
        completion = """<submit_program>
import pymc as pm
x = 1
</submit_program>"""
        thinking, program = parse_completion(completion)

        assert thinking is None
        assert "import pymc" in program

    def test_multiple_code_blocks_takes_first(self):
        completion = """```python
first_code = True
```

```python
second_code = True
```"""
        thinking, program = parse_completion(completion)

        assert "first_code" in program
        assert "second_code" not in program

    def test_empty_think_tags(self):
        completion = """<think></think>
```python
x = 1
```"""
        thinking, program = parse_completion(completion)

        assert thinking == ""
        assert "x = 1" in program

    def test_multiline_think_content(self):
        completion = """<think>
Line 1
Line 2
Line 3
</think>

```python
pass
```"""
        thinking, program = parse_completion(completion)

        assert "Line 1" in thinking
        assert "Line 2" in thinking
        assert "Line 3" in thinking


class TestLoadSFTJsonl:
    def test_load_valid_jsonl(self, tmp_path):
        jsonl_file = tmp_path / "train.jsonl"
        examples = [
            {
                "prompt": "Question 1",
                "completion": "```python\nx = 1\n```",
            },
            {
                "prompt": "Question 2",
                "completion": "<think>Thinking</think>\n```python\ny = 2\n```",
            },
        ]
        with open(jsonl_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        loaded = load_sft_jsonl(jsonl_file)

        assert len(loaded) == 2
        assert loaded[0].prompt == "Question 1"
        assert "x = 1" in loaded[0].program
        assert loaded[1].thinking is not None
        assert "Thinking" in loaded[1].thinking

    def test_skip_examples_without_program(self, tmp_path):
        jsonl_file = tmp_path / "train.jsonl"
        examples = [
            {"prompt": "Q1", "completion": "No code here"},
            {"prompt": "Q2", "completion": "```python\nx = 1\n```"},
        ]
        with open(jsonl_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        loaded = load_sft_jsonl(jsonl_file, require_program=True)

        assert len(loaded) == 1
        assert "x = 1" in loaded[0].program

    def test_include_examples_without_program(self, tmp_path):
        jsonl_file = tmp_path / "train.jsonl"
        examples = [
            {"prompt": "Q1", "completion": "No code here"},
            {"prompt": "Q2", "completion": "```python\nx = 1\n```"},
        ]
        with open(jsonl_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        loaded = load_sft_jsonl(jsonl_file, require_program=False)

        assert len(loaded) == 2

    def test_max_examples_limit(self, tmp_path):
        jsonl_file = tmp_path / "train.jsonl"
        examples = [
            {"prompt": f"Q{i}", "completion": f"```python\nx = {i}\n```"} for i in range(10)
        ]
        with open(jsonl_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        loaded = load_sft_jsonl(jsonl_file, max_examples=3)

        assert len(loaded) == 3

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_sft_jsonl(Path("/nonexistent/file.jsonl"))

    def test_skip_invalid_json_lines(self, tmp_path):
        jsonl_file = tmp_path / "train.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('{"prompt": "Q1", "completion": "```python\\nx = 1\\n```"}\n')
            f.write("not valid json\n")
            f.write('{"prompt": "Q2", "completion": "```python\\ny = 2\\n```"}\n')

        loaded = load_sft_jsonl(jsonl_file)

        assert len(loaded) == 2

    def test_skip_missing_keys(self, tmp_path):
        jsonl_file = tmp_path / "train.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('{"prompt": "Q1"}\n')  # missing completion
            f.write('{"prompt": "Q2", "completion": "```python\\nx = 1\\n```"}\n')

        loaded = load_sft_jsonl(jsonl_file)

        assert len(loaded) == 1

    def test_source_file_tracking(self, tmp_path):
        jsonl_file = tmp_path / "train.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('{"prompt": "Q1", "completion": "```python\\nx = 1\\n```"}\n')

        loaded = load_sft_jsonl(jsonl_file)

        assert loaded[0].source_file == str(jsonl_file)
        assert loaded[0].line_number == 1


class TestSFTToBufferEntry:
    def test_basic_conversion(self):
        example = SFTExample(
            prompt="Test prompt",
            completion="<think>thinking</think>\n```python\nx = 1\n```",
            thinking="thinking",
            program="x = 1",
        )

        entry = sft_to_buffer_entry(
            example,
            log_reward=-2.0,
            strip_thinking=True,
        )

        assert entry.log_reward == -2.0
        assert len(entry.actions) == 1
        assert entry.actions[0]["type"] == "submit_program"
        assert entry.actions[0]["payload"] == "x = 1"
        assert entry.observations == ["Test prompt"]

    def test_strip_thinking_true(self):
        example = SFTExample(
            prompt="P",
            completion="<think>long reasoning</think>\n```python\nx = 1\n```",
            thinking="long reasoning",
            program="x = 1",
        )

        entry = sft_to_buffer_entry(
            example,
            log_reward=0.0,
            strip_thinking=True,
        )

        assert entry.actions[0]["payload"] == "x = 1"

    def test_strip_thinking_false(self):
        example = SFTExample(
            prompt="P",
            completion="<think>reasoning</think>\n```python\nx = 1\n```",
            thinking="reasoning",
            program="x = 1",
        )

        entry = sft_to_buffer_entry(
            example,
            log_reward=0.0,
            strip_thinking=False,
        )

        assert entry.actions[0]["payload"] == "<think>reasoning</think>\n```python\nx = 1\n```"

    def test_default_preserves_full_completion(self):
        example = SFTExample(
            prompt="P",
            completion="<think>reasoning</think>\n```python\nx = 1\n```",
            thinking="reasoning",
            program="x = 1",
        )

        entry = sft_to_buffer_entry(
            example,
            log_reward=0.0,
        )

        assert entry.actions[0]["payload"] == example.completion

    def test_policy_version_metadata(self):
        example = SFTExample(
            prompt="P",
            completion="```python\nx = 1\n```",
            thinking=None,
            program="x = 1",
        )

        entry = sft_to_buffer_entry(
            example,
            log_reward=0.0,
            policy_version=0,
        )

        assert entry.policy_version == 0

    def test_log_reward_is_required(self):
        example = SFTExample(
            prompt="P",
            completion="```python\nx = 1\n```",
            thinking=None,
            program="x = 1",
        )

        # log_reward is a required keyword argument - omitting it should raise
        with pytest.raises(TypeError, match="log_reward"):
            sft_to_buffer_entry(example)

    def test_log_reward_none_raises_type_error(self):
        example = SFTExample(
            prompt="P",
            completion="```python\nx = 1\n```",
            thinking=None,
            program="x = 1",
        )

        with pytest.raises(TypeError, match="log_reward must be a float"):
            sft_to_buffer_entry(example, log_reward=None)

    def test_log_reward_explicit_value_works(self):
        example = SFTExample(
            prompt="P",
            completion="```python\nx = 1\n```",
            thinking=None,
            program="x = 1",
        )

        entry = sft_to_buffer_entry(example, log_reward=-5.0)
        assert entry.log_reward == -5.0


class TestSFTExampleProperties:
    def test_has_thinking_true(self):
        ex = SFTExample(
            prompt="P",
            completion="C",
            thinking="Some reasoning",
            program="x = 1",
        )

        assert ex.has_thinking is True

    def test_has_thinking_false_none(self):
        ex = SFTExample(
            prompt="P",
            completion="C",
            thinking=None,
            program="x = 1",
        )

        assert ex.has_thinking is False

    def test_has_thinking_false_empty(self):
        ex = SFTExample(
            prompt="P",
            completion="C",
            thinking="   ",
            program="x = 1",
        )

        assert ex.has_thinking is False
