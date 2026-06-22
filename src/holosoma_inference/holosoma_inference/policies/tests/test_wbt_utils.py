"""
Unit tests for holosoma_inference.policies.wbt_utils module.

Tests focus on timing logic for MotionClockUtil and TimestepUtil classes.
"""

from unittest import mock

from defusedxml import ElementTree

from holosoma_inference.policies.wbt_utils import MotionClockUtil, PinocchioRobot, TimestepUtil


class TestMotionClockUtil:
    """Test cases for MotionClockUtil class."""

    def test_elapsed_ms_first_call_returns_zero(self):
        """First call to elapsed_ms should return 0 (anchor is set to current time)."""
        mock_clock_sub = mock.Mock()
        mock_clock_sub.get_clock.return_value = 100

        clock_util = MotionClockUtil(mock_clock_sub)
        result = clock_util.elapsed_ms()

        assert result == 0  # 100 - 100 = 0

    def test_elapsed_ms_normal_progression(self):
        """Normal clock progression should return correct elapsed time."""
        mock_clock_sub = mock.Mock()
        mock_clock_sub.get_clock.side_effect = [100, 150, 200]

        clock_util = MotionClockUtil(mock_clock_sub)

        assert clock_util.elapsed_ms() == 0  # Anchor at 100: 100-100=0
        assert clock_util.elapsed_ms() == 50  # 150-100=50
        assert clock_util.elapsed_ms() == 100  # 200-100=100

    def test_elapsed_ms_handles_backward_clock_jump(self):
        """Backward clock jump (e.g., sim reset) should preserve elapsed progress."""
        mock_clock_sub = mock.Mock()
        # Clock sequence: 100 -> 200 -> 50 (jump back)
        mock_clock_sub.get_clock.side_effect = [100, 200, 50]

        clock_util = MotionClockUtil(mock_clock_sub)

        assert clock_util.elapsed_ms() == 0  # Anchor at 100: 100-100=0
        assert clock_util.elapsed_ms() == 100  # Normal: 200-100=100
        # After jump: elapsed_at_anchor becomes 100, new anchor=50
        # Result: 100 + (50-50) = 100
        assert clock_util.elapsed_ms() == 100

    def test_elapsed_ms_backward_jump_continues_correctly(self):
        """After backward jump, subsequent calls should continue from new anchor."""
        mock_clock_sub = mock.Mock()
        # Clock: 100 -> 200 -> 50 (jump) -> 80 -> 120
        mock_clock_sub.get_clock.side_effect = [100, 200, 50, 80, 120]

        clock_util = MotionClockUtil(mock_clock_sub)

        assert clock_util.elapsed_ms() == 0  # Anchor at 100
        assert clock_util.elapsed_ms() == 100  # 200-100
        assert clock_util.elapsed_ms() == 100  # Jump: 100 + (50-50)
        assert clock_util.elapsed_ms() == 130  # 100 + (80-50)
        assert clock_util.elapsed_ms() == 170  # 100 + (120-50)

    def test_elapsed_ms_logs_warning_on_backward_jump(self):
        """Backward clock jump should log a warning when logger is provided."""
        mock_clock_sub = mock.Mock()
        mock_clock_sub.get_clock.side_effect = [100, 200, 50]
        mock_logger = mock.Mock()

        clock_util = MotionClockUtil(mock_clock_sub)

        clock_util.elapsed_ms(log=mock_logger)  # Anchor
        clock_util.elapsed_ms(log=mock_logger)  # Normal
        clock_util.elapsed_ms(log=mock_logger)  # Jump back

        mock_logger.warning.assert_called_once_with("Clock jumped back; re-anchoring.")

    def test_reset_clears_state_and_calls_clock_sub_reset(self):
        """Reset should clear internal state and call clock_sub.reset_origin()."""
        mock_clock_sub = mock.Mock()
        mock_clock_sub.get_clock.side_effect = [100, 200, 0, 50]

        clock_util = MotionClockUtil(mock_clock_sub)

        # Build up some state
        assert clock_util.elapsed_ms() == 0
        assert clock_util.elapsed_ms() == 100

        # Reset
        clock_util.reset()

        # Verify reset_origin was called
        mock_clock_sub.reset_origin.assert_called_once()

        # After reset, should start fresh
        assert clock_util.elapsed_ms() == 0  # New anchor at 0
        assert clock_util.elapsed_ms() == 50  # 50-0=50

    def test_reset_clears_elapsed_at_anchor(self):
        """Reset should clear the accumulated elapsed_ms_at_anchor."""
        mock_clock_sub = mock.Mock()
        # Sequence: 100 -> 200 -> 50 (jump, accumulates 100) -> reset -> 0 -> 30
        mock_clock_sub.get_clock.side_effect = [100, 200, 50, 0, 30]

        clock_util = MotionClockUtil(mock_clock_sub)

        clock_util.elapsed_ms()  # Anchor at 100
        clock_util.elapsed_ms()  # 100ms elapsed
        clock_util.elapsed_ms()  # Jump back, elapsed_at_anchor=100

        clock_util.reset()

        # After reset, elapsed_at_anchor should be 0
        assert clock_util.elapsed_ms() == 0  # Fresh anchor at 0
        assert clock_util.elapsed_ms() == 30  # 30-0=30, not 130


class TestTimestepUtil:
    """Test cases for TimestepUtil class."""

    def test_get_timestep_normal_progression(self):
        """Normal timestep progression based on elapsed time."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.side_effect = [0, 20, 40, 60]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=0)

        assert timestep_util.get_timestep() == 0  # 0/20 = 0
        assert timestep_util.get_timestep() == 1  # 20/20 = 1
        assert timestep_util.get_timestep() == 2  # 40/20 = 2
        assert timestep_util.get_timestep() == 3  # 60/20 = 3

    def test_get_timestep_with_start_timestep(self):
        """Timesteps should be offset by start_timestep."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.side_effect = [0, 20, 40]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=10)

        assert timestep_util.get_timestep() == 10  # 0/20 + 10 = 10
        assert timestep_util.get_timestep() == 11  # 20/20 + 10 = 11
        assert timestep_util.get_timestep() == 12  # 40/20 + 10 = 12

    def test_get_timestep_handles_forward_jump_at_start(self):
        """Forward jump at start should trigger reset to prevent frame skipping."""
        mock_clock = mock.Mock()
        # Simulate realistic clock behavior:
        # - First call: 500ms elapsed (jumped ahead, > 1 step, triggers reset)
        # - After reset() re-anchors, subsequent elapsed values are relative to new anchor
        # - Values 10, 30 represent time since the re-anchor point
        mock_clock.elapsed_ms.side_effect = [500, 10, 30]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=0)

        result = timestep_util.get_timestep()
        assert result == 0  # Should return start_timestep, not jump to step 25
        mock_clock.reset.assert_called_once()

        # After reset re-anchors the clock, elapsed times are small again
        assert timestep_util.get_timestep() == 0  # 10/20 = 0 steps
        assert timestep_util.get_timestep() == 1  # 30/20 = 1 step

    def test_get_timestep_forward_jump_only_at_start(self):
        """Forward jump detection only applies when at start_timestep."""
        mock_clock = mock.Mock()
        # Start normal, then jump (should NOT reset since we're past start)
        mock_clock.elapsed_ms.side_effect = [0, 20, 500]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=0)

        assert timestep_util.get_timestep() == 0  # At start
        assert timestep_util.get_timestep() == 1  # Past start
        assert timestep_util.get_timestep() == 25  # Jump allowed (500/20=25)

        mock_clock.reset.assert_not_called()

    def test_get_timestep_logs_warning_on_forward_jump(self):
        """Forward jump at start should log a warning when logger is provided."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.return_value = 500
        mock_logger = mock.Mock()

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=0)
        timestep_util.get_timestep(log=mock_logger)

        mock_logger.warning.assert_called_once_with("Clock jumped ahead at start; re-anchoring.")

    def test_reset_without_start_timestep(self):
        """Reset without parameter should use original start_timestep."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.side_effect = [0, 20, 0, 20]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=5)

        assert timestep_util.get_timestep() == 5  # 0/20 + 5 = 5
        assert timestep_util.get_timestep() == 6  # 20/20 + 5 = 6

        timestep_util.reset()  # No parameter
        mock_clock.reset.assert_called_once()

        assert timestep_util.get_timestep() == 5  # Back to original start_timestep
        assert timestep_util.get_timestep() == 6

    def test_reset_with_new_start_timestep(self):
        """Reset with parameter should update start_timestep."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.side_effect = [0, 20, 0, 20]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=5)

        assert timestep_util.get_timestep() == 5  # 0/20 + 5 = 5
        assert timestep_util.get_timestep() == 6  # 20/20 + 5 = 6

        timestep_util.reset(start_timestep=100)  # New start
        mock_clock.reset.assert_called_once()

        assert timestep_util.get_timestep() == 100  # 0/20 + 100 = 100
        assert timestep_util.get_timestep() == 101  # 20/20 + 100 = 101

    def test_timestep_property_returns_cached_value(self):
        """The timestep property should return cached value without clock update."""
        mock_clock = mock.Mock()
        # Use 20ms which is exactly 1 step - won't trigger forward jump detection
        mock_clock.elapsed_ms.return_value = 20

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=0)

        # Before any get_timestep call, should return start_timestep
        assert timestep_util.timestep == 0

        # After get_timestep, should return updated value
        timestep_util.get_timestep()
        assert timestep_util.timestep == 1

        # Property access should not call elapsed_ms again
        call_count = mock_clock.elapsed_ms.call_count
        _ = timestep_util.timestep
        assert mock_clock.elapsed_ms.call_count == call_count

    def test_timestep_calculation_with_fractional_interval(self):
        """Timestep calculation should handle fractional intervals correctly."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.side_effect = [0, 33, 66, 100]

        timestep_util = TimestepUtil(mock_clock, interval_ms=33.33, start_timestep=0)

        assert timestep_util.get_timestep() == 0  # 0/33.33 = 0
        assert timestep_util.get_timestep() == 0  # 33/33.33 = 0 (floor)
        assert timestep_util.get_timestep() == 1  # 66/33.33 = 1 (floor)
        assert timestep_util.get_timestep() == 3  # 100/33.33 = 3 (floor)


class TestCreateXmlFromUrdfJointPruning:
    """`_create_xml_from_urdf` must prune movable joints absent from dof_names
    (e.g. the yam gripper's prismatic finger jaws) plus their child-link subtree,
    so Pinocchio sees exactly the actuated DOF. The yam URDF embeds 33 movable
    joints into ONNX `robot_urdf` metadata; without pruning the njoints==29 assert
    in PinocchioRobot.__init__ fires for every yam checkpoint at eval/deploy."""

    # A minimal URDF: 2 revolute "arm" joints + 1 prismatic "jaw" (mimic-like
    # extra DOF) hanging off the last link, plus a fixed sensor joint that MUST
    # be kept (Pinocchio absorbs fixed joints; they anchor tracked frames).
    _URDF = """<?xml version="1.0"?>
<robot name="t">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <link name="sensor"/>
  <link name="jaw"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <visual><geometry><box size="1 1 1"/></geometry></visual></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/></joint>
  <joint name="ft_fixed" type="fixed"><parent link="l2"/><child link="sensor"/></joint>
  <joint name="jaw_joint" type="prismatic"><parent link="sensor"/><child link="jaw"/></joint>
</robot>"""

    @staticmethod
    def _names(root, kind):
        return {
            e.get("name")
            for e in root
            if e.tag.rsplit("}", 1)[-1] == kind
        }

    def test_prunes_movable_joint_not_in_keep_and_its_link(self):
        out = PinocchioRobot._create_xml_from_urdf(self._URDF, keep_joint_names=["j1", "j2"])
        root = ElementTree.fromstring(out)
        joints = self._names(root, "joint")
        links = self._names(root, "link")
        # Movable jaw_joint dropped; its child link "jaw" dropped.
        assert "jaw_joint" not in joints
        assert "jaw" not in links
        # Kept joints + the fixed sensor joint survive.
        assert {"j1", "j2", "ft_fixed"} <= joints
        assert {"base", "l1", "l2", "sensor"} <= links

    def test_fixed_joints_always_kept(self):
        """A fixed joint is kept even when not in keep_joint_names (Pinocchio
        absorbs it; it may anchor a tracked foot/ref frame)."""
        out = PinocchioRobot._create_xml_from_urdf(self._URDF, keep_joint_names=["j1", "j2"])
        assert "ft_fixed" in self._names(ElementTree.fromstring(out), "joint")

    def test_noop_when_all_movable_kept(self):
        """When every movable joint is in keep_joint_names (rubber/pika case),
        no joints/links are pruned — only visuals/collisions are stripped."""
        keep = ["j1", "j2", "jaw_joint"]
        out = PinocchioRobot._create_xml_from_urdf(self._URDF, keep_joint_names=keep)
        root = ElementTree.fromstring(out)
        assert self._names(root, "joint") == {"j1", "j2", "ft_fixed", "jaw_joint"}
        assert self._names(root, "link") == {"base", "l1", "l2", "sensor", "jaw"}

    def test_none_keep_strips_only_visuals(self):
        """keep_joint_names=None preserves the legacy behavior: strip
        visual/collision, keep all joints/links."""
        out = PinocchioRobot._create_xml_from_urdf(self._URDF, keep_joint_names=None)
        root = ElementTree.fromstring(out)
        assert self._names(root, "joint") == {"j1", "j2", "ft_fixed", "jaw_joint"}
        # Visual under j1 was stripped.
        for j in root:
            if j.get("name") == "j1":
                assert j.find("visual") is None
