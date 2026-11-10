from functools import partial

import numpy as np
import pytest

from copy_cat.storage.trajectory_store import TrajectoryStore
from copy_cat.preprocessing import observation_preprocessor
from copy_cat.preprocessing.observation_preprocessor import (
    GRAYSCALE_WEIGHTS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ObservationPreprocessor,
)
from copy_cat.data.trajectory import Trajectory

OP = ObservationPreprocessor


def make_frame(shape=(8, 8, 3), dtype=np.uint8, seed=0):
    """Build a frame that looks like a captured screen observation."""
    rng = np.random.default_rng(seed)

    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 255, size=shape, dtype=dtype)

    return rng.random(size=shape).astype(dtype)


class TestColour:
    def test_grayscale_uses_luma_weights_and_keeps_dtype(self):
        frame = make_frame()

        gray = OP.to_grayscale(frame)

        expected = np.rint(frame.astype(np.float32) @ np.asarray(GRAYSCALE_WEIGHTS))

        assert gray.shape == (8, 8)
        assert gray.dtype == np.uint8
        assert np.array_equal(gray, expected.astype(np.uint8))

    def test_grayscale_keepdims_keeps_a_channel_axis(self):
        gray = OP.to_grayscale(make_frame(), keepdims=True)

        assert gray.shape == (8, 8, 1)

    def test_grayscale_passes_an_already_gray_frame_through(self):
        frame = make_frame(shape=(8, 8))

        assert np.array_equal(OP.to_grayscale(frame), frame)
        assert OP.to_grayscale(frame, keepdims=True).shape == (8, 8, 1)

    def test_grayscale_ignores_alpha(self):
        frame = make_frame(shape=(8, 8, 4))

        assert np.array_equal(OP.to_grayscale(frame), OP.to_grayscale(frame[:, :, :3]))

    def test_grayscale_of_a_float_frame_stays_float(self):
        gray = OP.to_grayscale(make_frame(dtype=np.float32))

        assert gray.dtype == np.float32

    def test_to_rgb_repeats_the_single_channel(self):
        frame = make_frame(shape=(8, 8))

        rgb = OP.to_rgb(frame)

        assert rgb.shape == (8, 8, 3)
        assert np.array_equal(rgb[:, :, 0], frame)
        assert np.array_equal(rgb[:, :, 2], frame)

    def test_to_rgb_leaves_a_three_channel_frame_alone(self):
        frame = make_frame()

        assert np.array_equal(OP.to_rgb(frame), frame)

    def test_swap_channels_reverses_rgb(self):
        frame = make_frame()

        assert np.array_equal(OP.swap_channels(frame), frame[:, :, ::-1])

    def test_swap_channels_keeps_alpha_last(self):
        frame = make_frame(shape=(8, 8, 4))

        swapped = OP.swap_channels(frame)

        assert np.array_equal(swapped[:, :, :3], frame[:, :, 2::-1])
        assert np.array_equal(swapped[:, :, 3], frame[:, :, 3])

    def test_drop_alpha_trims_the_fourth_channel(self):
        frame = make_frame(shape=(8, 8, 4))

        assert OP.drop_alpha(frame).shape == (8, 8, 3)
        assert np.array_equal(OP.drop_alpha(frame), frame[:, :, :3])

    def test_drop_alpha_leaves_a_three_channel_frame_alone(self):
        frame = make_frame()

        assert np.array_equal(OP.drop_alpha(frame), frame)

    @pytest.mark.parametrize("preprocessor", [OP.to_grayscale, OP.to_rgb, OP.swap_channels])
    def test_rejects_an_unexpected_channel_count(self, preprocessor):
        with pytest.raises(ValueError, match="channels"):
            preprocessor(make_frame(shape=(8, 8, 5)))


class TestGeometry:
    def test_crop_takes_the_requested_box(self):
        frame = make_frame(shape=(10, 12, 3))

        cropped = OP.crop(frame, top=2, left=3, height=4, width=5)

        assert cropped.shape == (4, 5, 3)
        assert np.array_equal(cropped, frame[2:6, 3:8])

    @pytest.mark.parametrize(
        "box",
        [
            (0, 0, 0, 4),  # zero height
            (0, 0, 4, -1),  # negative width
            (-1, 0, 4, 4),  # negative top
            (6, 0, 4, 4),  # runs off the bottom
            (0, 9, 4, 4),  # runs off the right
        ],
    )
    def test_crop_rejects_a_box_outside_the_frame(self, box):
        top, left, height, width = box

        with pytest.raises(ValueError):
            OP.crop(make_frame(), top, left, height, width)

    def test_center_crop_is_centred(self):
        frame = make_frame(shape=(10, 10, 3))

        cropped = OP.center_crop(frame, size=(4, 4))

        assert np.array_equal(cropped, frame[3:7, 3:7])

    def test_center_crop_rejects_a_size_larger_than_the_frame(self):
        with pytest.raises(ValueError, match="center-crop"):
            OP.center_crop(make_frame(), size=(16, 16))

    def test_downsample_subsamples_by_stride(self):
        frame = make_frame(shape=(8, 8, 3))

        assert np.array_equal(OP.downsample(frame, 2), frame[::2, ::2])

    def test_downsample_mean_averages_each_block(self):
        frame = np.arange(16, dtype=np.uint8).reshape(4, 4)

        averaged = OP.downsample(frame, 2, mode="mean")

        assert averaged.shape == (2, 2)
        assert averaged.dtype == np.uint8
        # The top-left block is 0, 1, 4, 5.
        assert averaged[0, 0] == round(np.mean([0, 1, 4, 5]))

    def test_downsample_mean_drops_a_partial_trailing_block(self):
        assert OP.downsample(make_frame(shape=(9, 9, 3)), 2, mode="mean").shape == (4, 4, 3)

    def test_downsample_by_one_is_a_no_op(self):
        frame = make_frame()

        assert np.array_equal(OP.downsample(frame, 1), frame)

    @pytest.mark.parametrize("factor,mode", [(0, "subsample"), (-2, "mean")])
    def test_downsample_rejects_a_non_positive_factor(self, factor, mode):
        with pytest.raises(ValueError, match="factor"):
            OP.downsample(make_frame(), factor, mode=mode)

    def test_downsample_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            OP.downsample(make_frame(), 2, mode="bilinear")

    def test_pad_centres_the_frame(self):
        frame = np.ones((2, 2), dtype=np.uint8)

        padded = OP.pad(frame, size=(4, 4), value=0)

        assert padded.shape == (4, 4)
        assert np.array_equal(padded[1:3, 1:3], frame)
        assert padded[0, 0] == 0

    def test_pad_can_anchor_at_the_corner(self):
        padded = OP.pad(np.ones((2, 2), dtype=np.uint8), size=(4, 4), center=False)

        assert np.array_equal(padded[0:2, 0:2], np.ones((2, 2), dtype=np.uint8))
        assert padded[3, 3] == 0

    def test_pad_keeps_the_channel_axis(self):
        assert OP.pad(make_frame(), size=(12, 12)).shape == (12, 12, 3)

    def test_pad_refuses_to_shrink(self):
        with pytest.raises(ValueError, match="shrink"):
            OP.pad(make_frame(), size=(4, 4))

    def test_flips_mirror_the_frame(self):
        frame = make_frame()

        assert np.array_equal(OP.flip_horizontal(frame), frame[:, ::-1])
        assert np.array_equal(OP.flip_vertical(frame), frame[::-1])


class FakeCv2:
    """Stands in for OpenCV so resize can be tested without depending on it."""

    INTER_AREA = 3
    INTER_LINEAR = 1
    INTER_NEAREST = 0
    INTER_CUBIC = 2

    def __init__(self):
        self.calls = []

    def resize(self, frame, size, interpolation=0):
        self.calls.append((frame.shape, size, interpolation))
        width, height = size
        # Nearest-neighbour is enough to check shapes, and mirrors OpenCV in dropping
        # a trailing axis of length one.
        rows = (np.arange(height) * frame.shape[0]) // height
        cols = (np.arange(width) * frame.shape[1]) // width
        resized = frame[rows[:, None], cols]

        if resized.ndim == 3 and resized.shape[2] == 1:
            return resized[:, :, 0]

        return resized


class TestResize:
    @pytest.fixture
    def fake(self, monkeypatch):
        fake = FakeCv2()
        monkeypatch.setattr(observation_preprocessor, "_import_cv2", lambda: fake)

        return fake

    def test_passes_width_height_to_opencv(self, fake):
        resized = OP.resize(make_frame(shape=(20, 40, 3)), size=(5, 10))

        assert resized.shape == (5, 10, 3)
        # size is (height, width) here but (width, height) for OpenCV.
        assert fake.calls == [((20, 40, 3), (10, 5), FakeCv2.INTER_AREA)]

    @pytest.mark.parametrize(
        "name,constant",
        [
            ("area", FakeCv2.INTER_AREA),
            ("linear", FakeCv2.INTER_LINEAR),
            ("nearest", FakeCv2.INTER_NEAREST),
            ("cubic", FakeCv2.INTER_CUBIC),
        ],
    )
    def test_maps_interpolation_names_to_opencv_constants(self, fake, name, constant):
        OP.resize(make_frame(), size=(4, 4), interpolation=name)

        assert fake.calls[0][2] == constant

    def test_rejects_an_unknown_interpolation(self, fake):
        with pytest.raises(ValueError, match="interpolation"):
            OP.resize(make_frame(), size=(4, 4), interpolation="lanczos")

    def test_restores_a_trailing_axis_opencv_dropped(self, fake):
        resized = OP.resize(make_frame(shape=(8, 8, 1)), size=(4, 4))

        assert resized.shape == (4, 4, 1)

    def test_keeps_a_grayscale_frame_two_dimensional(self, fake):
        assert OP.resize(make_frame(shape=(8, 8)), size=(4, 4)).shape == (4, 4)

    def test_rejects_more_channels_than_opencv_handles(self, fake):
        with pytest.raises(ValueError, match="channels"):
            OP.resize(make_frame(shape=(8, 8, 8)), size=(4, 4))

    @pytest.mark.parametrize("size", [(0, 4), (4, -1), 4, (1, 2, 3)])
    def test_rejects_a_bad_size(self, fake, size):
        with pytest.raises(ValueError, match="size"):
            OP.resize(make_frame(), size=size)

    def test_opencv_is_only_needed_for_resize(self, monkeypatch):
        def missing():
            raise ImportError(
                "Resizing an observation needs OpenCV, which is an optional dependency. "
                "Install it with: pip install 'copy-cat[replay]', or use downsample "
                "instead."
            )

        monkeypatch.setattr(observation_preprocessor, "_import_cv2", missing)

        frame = make_frame()

        with pytest.raises(ImportError, match=r"copy-cat\[replay\]"):
            OP.resize(frame, size=(4, 4))

        # Everything else still works without it.
        assert OP.downsample(OP.to_grayscale(frame), 2).shape == (4, 4)


class TestValues:
    def test_normalize_scales_uint8_into_the_unit_range(self):
        frame = np.array([[0, 128, 255]], dtype=np.uint8)

        normalized = OP.normalize(frame)

        assert normalized.dtype == np.float32
        assert normalized[0, 0] == 0.0
        assert normalized[0, 2] == pytest.approx(1.0)
        assert normalized[0, 1] == pytest.approx(128 / 255)

    def test_normalize_of_a_float_frame_is_a_cast(self):
        frame = make_frame(dtype=np.float64)

        normalized = OP.normalize(frame)

        assert normalized.dtype == np.float32
        assert normalized == pytest.approx(frame.astype(np.float32))

    def test_normalize_honours_an_explicit_max_value(self):
        frame = np.array([[0, 5, 10]], dtype=np.uint8)

        assert OP.normalize(frame, max_value=10)[0, 2] == pytest.approx(1.0)

    def test_normalize_rejects_a_non_positive_max_value(self):
        with pytest.raises(ValueError, match="max_value"):
            OP.normalize(make_frame(), max_value=0)

    def test_standardize_centres_each_channel(self):
        frame = np.zeros((2, 2, 3), dtype=np.float32)

        standardized = OP.standardize(frame)

        for channel in range(3):
            expected = -IMAGENET_MEAN[channel] / IMAGENET_STD[channel]

            assert standardized[:, :, channel] == pytest.approx(expected)

    def test_standardize_accepts_a_scalar_for_a_gray_frame(self):
        frame = np.full((2, 2), 0.5, dtype=np.float32)

        assert OP.standardize(frame, mean=0.5, std=0.25) == pytest.approx(0.0)

    def test_standardize_rejects_per_channel_stats_on_a_gray_frame(self):
        with pytest.raises(ValueError, match="grayscale"):
            OP.standardize(make_frame(shape=(8, 8), dtype=np.float32))

    def test_standardize_rejects_a_stat_per_wrong_channel_count(self):
        with pytest.raises(ValueError, match="one value per channel"):
            OP.standardize(make_frame(shape=(8, 8, 4), dtype=np.float32))

    def test_standardize_rejects_a_zero_std(self):
        with pytest.raises(ValueError, match="std"):
            OP.standardize(make_frame(dtype=np.float32), mean=0.0, std=0.0)

    def test_scale_to_range_spans_the_requested_range(self):
        frame = np.array([[0, 255]], dtype=np.uint8)

        scaled = OP.scale_to_range(frame)

        assert scaled[0, 0] == pytest.approx(-1.0)
        assert scaled[0, 1] == pytest.approx(1.0)

    def test_scale_to_range_rejects_an_inverted_range(self):
        with pytest.raises(ValueError, match="low"):
            OP.scale_to_range(make_frame(), low=1.0, high=-1.0)

    def test_to_uint8_undoes_normalize(self):
        frame = make_frame()

        assert np.array_equal(OP.to_uint8(OP.normalize(frame)), frame)

    def test_to_uint8_clips_out_of_range_values(self):
        frame = np.array([[-0.5, 0.5, 1.5]], dtype=np.float32)

        assert np.array_equal(OP.to_uint8(frame), np.array([[0, 128, 255]], np.uint8))

    def test_to_uint8_reads_an_explicit_source_range(self):
        frame = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

        rescaled = OP.to_uint8(frame, low=-1.0, high=1.0)

        assert np.array_equal(rescaled, np.array([[0, 128, 255]], np.uint8))

    def test_clip_keeps_the_dtype(self):
        frame = make_frame()

        clipped = OP.clip(frame, 50, 200)

        assert clipped.dtype == np.uint8
        assert clipped.min() >= 50
        assert clipped.max() <= 200

    def test_clip_rejects_an_inverted_range(self):
        with pytest.raises(ValueError, match="low"):
            OP.clip(make_frame(), 200, 50)

    def test_binarize_produces_two_values(self):
        frame = np.array([[0, 127, 128, 255]], dtype=np.uint8)

        mask = OP.binarize(frame, threshold=127)

        assert mask.dtype == np.uint8
        assert np.array_equal(mask, np.array([[0, 0, 255, 255]], dtype=np.uint8))

    def test_binarize_honours_an_explicit_high(self):
        frame = np.array([[0, 255]], dtype=np.uint8)

        assert np.array_equal(OP.binarize(frame, 127, high=1), np.array([[0, 1]], np.uint8))

    def test_quantize_collapses_onto_levels(self):
        frame = np.arange(0, 256, dtype=np.uint8).reshape(16, 16)

        quantized = OP.quantize(frame, levels=2)

        assert quantized.dtype == np.uint8
        assert set(np.unique(quantized).tolist()) == {0, 255}

    def test_quantize_keeps_the_endpoints(self):
        frame = np.array([[0, 255]], dtype=np.uint8)

        quantized = OP.quantize(frame, levels=4)

        assert np.array_equal(quantized, frame)
        assert len(np.unique(OP.quantize(make_frame(), levels=4))) <= 4

    def test_quantize_rejects_too_few_levels(self):
        with pytest.raises(ValueError, match="levels"):
            OP.quantize(make_frame(), levels=1)


class TestLayout:
    def test_channels_first_and_last_round_trip(self):
        frame = make_frame()

        first = OP.to_channels_first(frame)

        assert first.shape == (3, 8, 8)
        assert np.array_equal(OP.to_channels_last(first), frame)

    def test_channels_first_adds_an_axis_to_a_gray_frame(self):
        assert OP.to_channels_first(make_frame(shape=(8, 8))).shape == (1, 8, 8)

    def test_channels_last_adds_an_axis_to_a_gray_frame(self):
        assert OP.to_channels_last(make_frame(shape=(8, 8))).shape == (8, 8, 1)

    def test_add_batch_dim_prepends_one(self):
        assert OP.add_batch_dim(make_frame()).shape == (1, 8, 8, 3)

    def test_flatten_collapses_to_one_dimension(self):
        frame = make_frame()

        flat = OP.flatten(frame)

        assert flat.shape == (8 * 8 * 3,)
        assert np.array_equal(flat, frame.reshape(-1))

    @pytest.mark.parametrize(
        "preprocessor", [OP.to_grayscale, OP.to_channels_first, OP.flip_horizontal]
    )
    @pytest.mark.parametrize("shape", [(8,), (2, 2, 2, 2, 2)])
    def test_rejects_a_rank_that_is_neither_a_frame_nor_a_batch(self, preprocessor, shape):
        with pytest.raises(ValueError, match="batch of frames"):
            preprocessor(np.zeros(shape, dtype=np.uint8))

    @pytest.mark.parametrize("preprocessor", [OP.to_grayscale, OP.normalize, OP.flatten])
    def test_rejects_a_non_array(self, preprocessor):
        with pytest.raises(TypeError, match="numpy array"):
            preprocessor([[1, 2], [3, 4]])


def make_batch(count=5, shape=(8, 8, 3), dtype=np.uint8):
    """Build an episode's worth of frames, the shape a loaded trajectory stacks into."""
    frames = [make_frame(shape=shape, dtype=dtype, seed=seed) for seed in range(count)]

    return np.stack(frames)


class TestTemporal:
    def test_framestack_repeats_a_single_frame(self):
        frame = make_frame(shape=(8, 8), seed=7)

        stacked = OP.framestack(frame, num_frames=4)

        assert stacked.shape == (8, 8, 4)

        for index in range(4):
            assert np.array_equal(stacked[:, :, index], frame)

    def test_framestack_can_bootstrap_a_single_frame_with_zeros(self):
        frame = make_frame(shape=(8, 8), seed=3)

        stacked = OP.framestack(frame, num_frames=4, fill="zeros")

        assert stacked.shape == (8, 8, 4)
        assert np.array_equal(stacked[:, :, :3], np.zeros((8, 8, 3), dtype=frame.dtype))
        assert np.array_equal(stacked[:, :, 3], frame)

    def test_framestack_slides_a_window_over_a_batch(self):
        batch = make_batch(shape=(8, 8, 1))

        stacked = OP.framestack(batch, num_frames=3)

        assert stacked.shape == (5, 8, 8, 3)

        # The last frame carries frames 2, 3 and 4, oldest first.
        for offset, source in enumerate((2, 3, 4)):
            assert np.array_equal(stacked[4, :, :, offset], batch[source, :, :, 0])

    def test_framestack_bootstraps_the_start_of_a_batch(self):
        batch = make_batch(shape=(8, 8, 1))

        repeated = OP.framestack(batch, num_frames=3)
        zeroed = OP.framestack(batch, num_frames=3, fill="zeros")

        # The first frame has no history: repeat pads with itself, zeros pads with black.
        for offset in range(3):
            assert np.array_equal(repeated[0, :, :, offset], batch[0, :, :, 0])

        assert np.array_equal(zeroed[0, :, :, :2], np.zeros((8, 8, 2), dtype=batch.dtype))
        assert np.array_equal(zeroed[0, :, :, 2], batch[0, :, :, 0])

    def test_framestack_keeps_the_channels_of_a_colour_batch(self):
        batch = make_batch(count=3)

        stacked = OP.framestack(batch, num_frames=2)

        assert stacked.shape == (3, 8, 8, 6)
        assert np.array_equal(stacked[2, :, :, :3], batch[1])
        assert np.array_equal(stacked[2, :, :, 3:], batch[2])

    def test_framestack_rejects_bad_arguments(self):
        with pytest.raises(ValueError, match="num_frames"):
            OP.framestack(make_frame(), num_frames=0)

        with pytest.raises(ValueError, match="fill"):
            OP.framestack(make_frame(), num_frames=4, fill="left")

    def test_stack_frames_joins_gray_frames_on_a_new_axis(self):
        frames = [make_frame(shape=(8, 8), seed=seed) for seed in range(4)]

        stacked = OP.stack_frames(frames)

        assert stacked.shape == (8, 8, 4)
        assert np.array_equal(stacked[:, :, 2], frames[2])

    def test_stack_frames_concatenates_channelled_frames(self):
        frames = [make_frame(seed=seed) for seed in range(2)]

        assert OP.stack_frames(frames).shape == (8, 8, 6)

    def test_stack_frames_rejects_an_empty_sequence(self):
        with pytest.raises(ValueError, match="at least one frame"):
            OP.stack_frames([])

    def test_stack_frames_rejects_mismatched_frames(self):
        frames = [make_frame(shape=(8, 8)), make_frame(shape=(4, 4))]

        with pytest.raises(ValueError, match="share one shape and dtype"):
            OP.stack_frames(frames)

    def test_frame_difference_does_not_wrap_around(self):
        current = np.array([[0]], dtype=np.uint8)
        previous = np.array([[255]], dtype=np.uint8)

        difference = OP.frame_difference(current, previous)

        assert difference.dtype == np.uint8
        assert difference[0, 0] == 255

    def test_frame_difference_can_keep_the_sign(self):
        current = np.array([[0]], dtype=np.uint8)
        previous = np.array([[255]], dtype=np.uint8)

        signed = OP.frame_difference(current, previous, signed=True)

        assert signed[0, 0] == -255

    def test_frame_difference_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            OP.frame_difference(make_frame(), make_frame(shape=(4, 4, 3)))


class TestFramestackStream:
    def test_repeats_the_first_frame_to_bootstrap(self):
        stage = OP.framestack_stream(num_frames=4, fill="repeat")
        frame = make_frame(shape=(8, 8), seed=7)

        stacked = stage(frame)

        assert stacked.shape == (8, 8, 4)

        for index in range(4):
            assert np.array_equal(stacked[:, :, index], frame)

    def test_can_bootstrap_with_zero_padding(self):
        stage = OP.framestack_stream(num_frames=4, fill="zeros")
        frame = make_frame(shape=(8, 8), seed=3)

        stacked = stage(frame)

        assert stacked.shape == (8, 8, 4)
        assert np.array_equal(stacked[:, :, :3], np.zeros((8, 8, 3), dtype=frame.dtype))
        assert np.array_equal(stacked[:, :, 3], frame)

    def test_slides_over_time(self):
        stage = OP.framestack_stream(num_frames=3)
        first = make_frame(shape=(8, 8), seed=1)
        second = make_frame(shape=(8, 8), seed=2)
        third = make_frame(shape=(8, 8), seed=3)

        stage(first)
        stage(second)
        stacked = stage(third)

        assert np.array_equal(stacked[:, :, 0], first)
        assert np.array_equal(stacked[:, :, 1], second)
        assert np.array_equal(stacked[:, :, 2], third)

    def test_reset_clears_internal_state(self):
        stage = OP.framestack_stream(num_frames=2)
        first = make_frame(shape=(8, 8), seed=11)
        second = make_frame(shape=(8, 8), seed=12)

        stage(first)
        stage(second)
        stage.reset()
        reset_output = stage(second)

        assert np.array_equal(reset_output[:, :, 0], second)
        assert np.array_equal(reset_output[:, :, 1], second)

    def test_matches_framestack_over_the_same_episode(self):
        """Streaming a recording must give what framestacking it afterwards would."""
        batch = make_batch(count=6, shape=(8, 8, 1))
        stage = OP.framestack_stream(num_frames=3)

        streamed = np.stack([stage(frame) for frame in batch])

        assert np.array_equal(streamed, OP.framestack(batch, num_frames=3))

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError, match="num_frames"):
            OP.framestack_stream(num_frames=0)

        with pytest.raises(ValueError, match="fill"):
            OP.framestack_stream(num_frames=4, fill="left")

    def test_rejects_mismatched_frames(self):
        stage = OP.framestack_stream(num_frames=2)
        stage(make_frame(shape=(8, 8), dtype=np.uint8, seed=0))

        with pytest.raises(ValueError, match="share one shape and dtype"):
            stage(make_frame(shape=(8, 8), dtype=np.float32, seed=1))

    def test_rejects_a_batch(self):
        stage = OP.framestack_stream(num_frames=2)

        with pytest.raises(ValueError, match="one frame at a time"):
            stage(make_batch(shape=(8, 8, 1)))


class TestCompose:
    def test_applies_preprocessors_in_order(self):
        frame = make_frame(shape=(16, 16, 3))

        pipeline = OP.compose(
            OP.to_grayscale,
            partial(OP.downsample, factor=2),
            OP.normalize,
        )
        observation = pipeline(frame)

        assert observation.shape == (8, 8)
        assert observation.dtype == np.float32
        assert observation.max() <= 1.0

    def test_an_empty_pipeline_is_the_identity(self):
        frame = make_frame()

        assert np.array_equal(OP.compose()(frame), frame)

    def test_rejects_a_non_callable(self):
        with pytest.raises(ValueError, match="callables"):
            OP.compose(OP.to_grayscale, "normalize")


FRAMESTACK = partial(OP.framestack, num_frames=2)

SINGLE_FRAME_PREPROCESSORS = [
    OP.to_grayscale,
    OP.to_rgb,
    OP.swap_channels,
    OP.drop_alpha,
    partial(OP.crop, top=1, left=1, height=4, width=4),
    partial(OP.center_crop, size=(4, 4)),
    partial(OP.downsample, factor=2),
    partial(OP.downsample, factor=2, mode="mean"),
    partial(OP.pad, size=(12, 12)),
    OP.flip_horizontal,
    OP.flip_vertical,
    OP.normalize,
    OP.standardize,
    OP.scale_to_range,
    OP.to_uint8,
    partial(OP.clip, low=10, high=200),
    partial(OP.binarize, threshold=100),
    partial(OP.quantize, levels=4),
    OP.to_channels_first,
    OP.to_channels_last,
    OP.add_batch_dim,
    OP.flatten,
    FRAMESTACK,
]

# to_grayscale is rebound because a batch always keeps its channel axis, and the two
# preprocessors whose whole job is the batch axis cannot match a frame-at-a-time run.
BATCH_PREPROCESSORS = [
    preprocessor
    for preprocessor in SINGLE_FRAME_PREPROCESSORS
    if preprocessor not in (OP.to_grayscale, OP.add_batch_dim, FRAMESTACK)
] + [partial(OP.to_grayscale, keepdims=True)]


@pytest.mark.parametrize("preprocessor", SINGLE_FRAME_PREPROCESSORS)
def test_no_preprocessor_writes_to_its_input(preprocessor):
    """Frames loaded from a store are read-only memmaps, so nothing may write in place."""
    frame = make_frame()
    frame.flags.writeable = False
    original = frame.copy()

    result = preprocessor(frame)

    assert np.array_equal(frame, original)

    # A pass-through may hand the input straight back; anything else must be a new array.
    if result.shape == frame.shape and result.dtype == frame.dtype:
        assert result is frame or not np.shares_memory(result, frame)


@pytest.mark.parametrize("preprocessor", SINGLE_FRAME_PREPROCESSORS)
def test_no_preprocessor_writes_to_a_read_only_batch(preprocessor):
    """A stacked episode is read-only too, since it comes off the same memmaps."""
    batch = make_batch()
    batch.flags.writeable = False
    original = batch.copy()

    result = preprocessor(batch)

    assert np.array_equal(batch, original)

    if result.shape == batch.shape and result.dtype == batch.dtype:
        assert result is batch or not np.shares_memory(result, batch)


@pytest.mark.parametrize("preprocessor", BATCH_PREPROCESSORS)
def test_a_batch_matches_processing_one_frame_at_a_time(preprocessor):
    """Batching is only worth having if it changes nothing but the number of calls."""
    batch = make_batch()

    assert np.array_equal(
        preprocessor(batch), np.stack([preprocessor(frame) for frame in batch])
    )


class TestBatch:
    def test_grayscale_keeps_a_batch_four_dimensional(self):
        assert OP.to_grayscale(make_batch()).shape == (5, 8, 8, 1)

    def test_flatten_gives_one_row_per_frame(self):
        assert OP.flatten(make_batch()).shape == (5, 8 * 8 * 3)

    def test_add_batch_dim_leaves_a_batch_alone(self):
        batch = make_batch()

        assert OP.add_batch_dim(batch) is batch

    def test_channels_first_and_last_round_trip_over_a_batch(self):
        batch = make_batch()

        first = OP.to_channels_first(batch)

        assert first.shape == (5, 3, 8, 8)
        assert np.array_equal(OP.to_channels_last(first), batch)

    def test_resize_handles_a_batch_one_frame_at_a_time(self, monkeypatch):
        fake = FakeCv2()
        monkeypatch.setattr(observation_preprocessor, "_import_cv2", lambda: fake)
        batch = make_batch(count=3, shape=(20, 40, 3))

        resized = OP.resize(batch, size=(5, 10))

        assert resized.shape == (3, 5, 10, 3)
        assert len(fake.calls) == 3

    @pytest.mark.parametrize("shape", [(8,), (2, 2, 2, 2, 2)])
    def test_rejects_a_rank_that_is_neither_a_frame_nor_a_batch(self, shape):
        with pytest.raises(ValueError, match="batch of frames"):
            OP.normalize(np.zeros(shape, dtype=np.uint8))

    def test_a_pipeline_runs_over_a_whole_episode(self):
        batch = make_batch(count=6, shape=(32, 32, 3))
        pipeline = OP.compose(
            OP.to_grayscale,
            partial(OP.downsample, factor=2, mode="mean"),
            partial(OP.center_crop, size=(8, 8)),
            partial(OP.framestack, num_frames=4),
            OP.normalize,
        )

        observations = pipeline(batch)

        assert observations.shape == (6, 8, 8, 4)
        assert observations.dtype == np.float32
        assert observations.max() <= 1.0


class TestStorageRoundTrip:
    def test_a_preprocessed_episode_still_saves_and_loads(self, tmp_path):
        pipeline = OP.compose(OP.to_grayscale, partial(OP.downsample, factor=2))
        trajectory = Trajectory()

        for step in range(3):
            trajectory.record_step(
                state=pipeline(make_frame(shape=(16, 16, 3), seed=step)),
                action=step,
                reward=None,
                timestamp=round(step * 0.1, 3),
            )

        store = TrajectoryStore(tmp_path)
        name = store.save(trajectory, fps=10)
        loaded = store.load(name)

        assert store.info(name)["obs_shape"] == [8, 8]
        assert store.info(name)["obs_dtype"] == "uint8"

        for step in range(3):
            assert np.array_equal(loaded.get_state(step), trajectory.get_state(step))

    def test_a_pipeline_runs_over_read_only_memmapped_frames(self, tmp_path):
        trajectory = Trajectory()

        for step in range(3):
            trajectory.record_step(
                state=make_frame(shape=(16, 16, 3), seed=step),
                action=step,
                reward=None,
            )

        store = TrajectoryStore(tmp_path)
        name = store.save(trajectory)
        loaded = store.load(name, mmap=True)

        stored = loaded.get_state(0)

        assert not stored.flags.writeable

        pipeline = OP.compose(OP.to_grayscale, partial(OP.downsample, factor=2), OP.normalize)

        assert pipeline(stored).shape == (8, 8)
