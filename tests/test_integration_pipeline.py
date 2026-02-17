"""End-to-end pipeline integration tests."""

import pytest
import torch


class TestPipelineIntegration:
    def test_task_to_loss_pipeline(self):
        """Full pipeline from task creation to loss computation."""
        import torch.nn as nn

        from synthstats.policies.hf_policy import MockPolicy
        from synthstats.runtime.codecs import JSONToolCodec
        from synthstats.runtime.rollout import RolloutConfig, rollout_episode
        from synthstats.train.objectives.losses import subtb_loss
        from tests.fixtures import ToyJudge, ToyTask

        policy = MockPolicy(
            fixed_text='{"answer": "42"}',
            fixed_token_ids=[1, 2, 3],
            fixed_token_logprobs=[-0.5, -0.3, -0.2],
        )
        task = ToyTask()
        codec = JSONToolCodec()
        judge = ToyJudge()

        cfg = RolloutConfig(max_steps=5)
        traj = rollout_episode(
            task=task,
            policy=policy,
            codec=codec,
            executors={},
            judge=judge,
            cfg=cfg,
        )

        all_logprobs = []
        all_mask = []
        for logprobs, mask in zip(traj.token_logprobs, traj.loss_mask, strict=False):
            all_logprobs.extend(logprobs)
            all_mask.extend(mask)

        if all_logprobs:
            log_probs = torch.tensor([all_logprobs])
            loss_mask = torch.tensor([all_mask], dtype=torch.bool)
            log_rewards = torch.tensor([traj.reward.total]).log()
            logZ = nn.Parameter(torch.tensor(0.0))

            loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

            assert loss.dim() == 0  # scalar
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_multi_trajectory_batch(self):
        """Multiple trajectories form a valid batch for loss."""
        import torch.nn as nn

        from synthstats.policies.hf_policy import MockPolicy
        from synthstats.runtime.codecs import JSONToolCodec
        from synthstats.runtime.rollout import RolloutConfig, rollout_episode
        from synthstats.train.objectives.losses import subtb_loss
        from tests.fixtures import ToyJudge, ToyTask

        policy = MockPolicy(
            fixed_text='{"answer": "x"}',
            fixed_token_ids=[1, 2, 3, 4],
            fixed_token_logprobs=[-0.1, -0.2, -0.3, -0.4],
        )
        cfg = RolloutConfig(max_steps=3)

        trajectories = []
        for _ in range(4):
            traj = rollout_episode(
                task=ToyTask(),
                policy=policy,
                codec=JSONToolCodec(),
                executors={},
                judge=ToyJudge(),
                cfg=cfg,
            )
            trajectories.append(traj)

        batch_logprobs = []
        batch_masks = []
        log_rewards = []

        for traj in trajectories:
            flat_lp = []
            flat_mask = []
            for lps, masks in zip(traj.token_logprobs, traj.loss_mask, strict=False):
                flat_lp.extend(lps)
                flat_mask.extend(masks)
            batch_logprobs.append(flat_lp)
            batch_masks.append(flat_mask)
            log_rewards.append(max(traj.reward.total, 1e-10))

        max_len = max(len(lp) for lp in batch_logprobs)
        for i in range(len(batch_logprobs)):
            pad_len = max_len - len(batch_logprobs[i])
            batch_logprobs[i].extend([0.0] * pad_len)
            batch_masks[i].extend([False] * pad_len)

        log_probs = torch.tensor(batch_logprobs)
        loss_mask = torch.tensor(batch_masks, dtype=torch.bool)
        log_rewards_t = torch.tensor(log_rewards).log()
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards_t, logZ)

        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestThoughtMaskIntegration:
    def test_find_think_blocks(self):
        """find_think_blocks detects thinking sections."""
        from synthstats.modeling.thought_mask import find_think_blocks

        text = "<think>Let me reason about this...</think>The answer is 42."

        blocks = find_think_blocks(text)

        assert len(blocks) == 1
        assert "reason" in blocks[0].content

    def test_multiple_think_blocks(self):
        """Multiple think blocks are detected."""
        from synthstats.modeling.thought_mask import find_think_blocks

        text = "<think>First thought</think>Middle<think>Second thought</think>End"

        blocks = find_think_blocks(text)

        assert len(blocks) == 2
        assert "First" in blocks[0].content
        assert "Second" in blocks[1].content

    def test_unclosed_think_block(self):
        """Unclosed think block extends to end of text."""
        from synthstats.modeling.thought_mask import find_think_blocks

        text = "Start<think>This is not closed"

        blocks = find_think_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].end_char == len(text)

    def test_clean_text_removes_think(self):
        """create_think_mask_simple removes think blocks."""
        from synthstats.modeling.thought_mask import create_think_mask_simple

        text = "<think>Let me reason about this...</think>The answer is 42."

        cleaned, spans = create_think_mask_simple(text)

        assert "think" not in cleaned.lower()
        assert "42" in cleaned
        assert len(spans) == 1

    def test_empty_text_no_blocks(self):
        """Empty text produces no blocks."""
        from synthstats.modeling.thought_mask import find_think_blocks

        blocks = find_think_blocks("")

        assert len(blocks) == 0

    def test_no_think_blocks(self):
        """Text without think blocks returns empty list."""
        from synthstats.modeling.thought_mask import find_think_blocks

        text = "Just regular text without any thinking tags."

        blocks = find_think_blocks(text)

        assert len(blocks) == 0

    def test_think_block_span_attributes(self):
        """ThinkBlockSpan has correct attributes."""
        from synthstats.modeling.thought_mask import find_think_blocks

        text = "abc<think>xyz</think>def"

        blocks = find_think_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].start_char == 3
        assert blocks[0].end_char == 21
        assert blocks[0].content == "xyz"


class TestBoxingCodecIntegration:
    def test_boxing_codec_parses_tool_call(self):
        """Boxing codec parses tool_call format."""
        from synthstats.core.types import ToolCall
        from synthstats.tasks.boxing.codecs import BoxingCodec

        codec = BoxingCodec()

        text = '<tool_call>{"name": "query", "input": {"x": 1.5}}</tool_call>'
        action = codec.parse(text)

        assert isinstance(action, ToolCall)
        assert action.name == "query"
        assert action.input == {"x": 1.5}

    def test_boxing_codec_parses_submit_program(self):
        """Boxing codec parses submit_program format."""
        from synthstats.core.types import Program
        from synthstats.tasks.boxing.codecs import BoxingCodec

        codec = BoxingCodec()

        text = "<submit_program>model = pm.Model()</submit_program>"
        action = codec.parse(text)

        assert isinstance(action, Program)
        assert "pm.Model" in action.code

    def test_boxing_codec_raises_for_invalid(self):
        """Boxing codec raises ParseError for unparseable text."""
        from synthstats.runtime.codecs import ParseError
        from synthstats.tasks.boxing.codecs import BoxingCodec

        codec = BoxingCodec()

        with pytest.raises(ParseError):
            codec.parse("This is just regular text with no action markup")

    def test_boxing_codec_format_tool_call(self):
        """Boxing codec formats ToolCall to text."""
        from synthstats.core.types import ToolCall
        from synthstats.tasks.boxing.codecs import BoxingCodec

        codec = BoxingCodec()
        action = ToolCall(name="query", input={"x": 1}, raw="")

        text = codec.render(action)

        assert "<tool_call>" in text
        assert '"name": "query"' in text

    def test_boxing_codec_render_program(self):
        """Boxing codec renders Program to text."""
        from synthstats.core.types import Program
        from synthstats.tasks.boxing.codecs import BoxingCodec

        codec = BoxingCodec()
        action = Program(code="print('hello')", language="pymc")

        text = codec.render(action)

        assert "<submit_program>" in text
        assert "print('hello')" in text


class TestJSONCodecIntegration:
    def test_json_codec_parses_tool(self):
        """JSONToolCodec parses tool JSON."""
        from synthstats.core.types import ToolCall
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()

        text = '```json\n{"tool": "search", "input": {"query": "test"}}\n```'
        action = codec.parse(text)

        assert isinstance(action, ToolCall)
        assert action.name == "search"

    def test_json_codec_parses_answer(self):
        """JSONToolCodec parses answer JSON."""
        from synthstats.core.types import FinalAnswer
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()

        text = '{"answer": "42"}'
        action = codec.parse(text)

        assert isinstance(action, FinalAnswer)
        assert action.text == "42"

    def test_json_codec_parses_program(self):
        """JSONToolCodec parses program JSON."""
        from synthstats.core.types import Program
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()

        text = '{"program": "import pymc as pm", "language": "pymc"}'
        action = codec.parse(text)

        assert isinstance(action, Program)
        assert action.code == "import pymc as pm"
        assert action.language == "pymc"

    def test_json_codec_parse_error(self):
        """JSONToolCodec raises ParseError for invalid input."""
        from synthstats.runtime.codecs import JSONToolCodec, ParseError

        codec = JSONToolCodec()

        with pytest.raises(ParseError):
            codec.parse("This is not JSON at all")


class TestSubTBLossIntegration:
    def test_subtb_loss_scalar_output(self):
        """SubTB loss returns scalar tensor."""
        import torch.nn as nn

        from synthstats.train.objectives.losses import subtb_loss

        log_probs = torch.tensor([[-0.5, -0.3, -0.2]])
        loss_mask = torch.ones(1, 3, dtype=torch.bool)
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.5))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert loss.dim() == 0
        assert loss.shape == torch.Size([])

    def test_subtb_loss_gradient_flows(self):
        """Gradient flows through SubTB loss to logZ."""
        import torch.nn as nn

        from synthstats.train.objectives.losses import subtb_loss

        log_probs = torch.tensor([[-0.5, -0.3]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)
        loss.backward()

        assert logZ.grad is not None
        assert logZ.grad != 0

    def test_subtb_loss_mask_excludes_tokens(self):
        """Loss mask excludes tokens from loss computation."""
        import torch.nn as nn

        from synthstats.train.objectives.losses import subtb_loss

        log_probs = torch.tensor([[-1.0, -1.0, -1.0]])
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        # all tokens included
        mask_all = torch.ones(1, 3, dtype=torch.bool)
        loss_all = subtb_loss(log_probs, mask_all, log_rewards, logZ)

        # only first token
        mask_one = torch.tensor([[True, False, False]])
        loss_one = subtb_loss(log_probs, mask_one, log_rewards, logZ)

        assert not torch.isnan(loss_all)
        assert not torch.isnan(loss_one)
        assert loss_all != loss_one

    def test_subtb_loss_handles_zero_reward(self):
        """SubTB loss handles zero reward (clamped to small positive)."""
        import torch.nn as nn

        from synthstats.train.objectives.losses import subtb_loss

        log_probs = torch.tensor([[-0.5, -0.3]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        # log of very small value (simulating near-zero reward)
        log_rewards = torch.tensor([-20.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_subtb_loss_batch_mean(self):
        """SubTB loss averages over batch dimension."""
        import torch.nn as nn

        from synthstats.train.objectives.losses import subtb_loss

        log_probs = torch.tensor([[-0.5, -0.3], [-0.2, -0.4]])
        loss_mask = torch.ones(2, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.0, 0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        # should be scalar (averaged over batch)
        assert loss.dim() == 0

    def test_subtb_loss_residual_clamping(self):
        """SubTB loss clamps extreme residuals."""
        import torch.nn as nn

        from synthstats.train.objectives.losses import subtb_loss

        # extreme log probs that would cause numerical issues
        log_probs = torch.tensor([[-100.0, -100.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ, max_residual=50.0)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestTrajectoryDatastructure:
    def test_trajectory_fields(self):
        """Trajectory has all required fields."""
        from synthstats.core.types import Message, Reward, Trajectory

        traj = Trajectory(
            messages=[Message(role="user", content="test")],
            token_ids=[[1, 2, 3]],
            token_logprobs=[[-0.1, -0.2, -0.3]],
            loss_mask=[[True, True, True]],
            reward=Reward(total=1.0, components={}, info={}),
        )

        assert len(traj.messages) == 1
        assert len(traj.token_ids) == 1
        assert len(traj.token_logprobs) == 1
        assert len(traj.loss_mask) == 1
        assert traj.reward.total == 1.0

    def test_trajectory_alignment(self):
        """Trajectory maintains alignment invariants."""
        from synthstats.core.types import Message, Reward, Trajectory

        # create multi-generation trajectory
        traj = Trajectory(
            messages=[
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="q2"),
                Message(role="assistant", content="a2"),
            ],
            token_ids=[[1, 2], [3, 4, 5]],
            token_logprobs=[[-0.1, -0.2], [-0.3, -0.4, -0.5]],
            loss_mask=[[True, True], [True, True, True]],
            reward=Reward(total=0.5, components={}, info={}),
        )

        # token data aligns (one entry per generation)
        assert len(traj.token_ids) == len(traj.token_logprobs)
        assert len(traj.token_ids) == len(traj.loss_mask)

        # each generation has aligned tokens/logprobs/mask
        for ids, lps, mask in zip(
            traj.token_ids, traj.token_logprobs, traj.loss_mask, strict=False
        ):
            assert len(ids) == len(lps) == len(mask)
