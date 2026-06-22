from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pinocchio as pin
from defusedxml import ElementTree

from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.misc import get_index_of_a_in_b

if TYPE_CHECKING:
    from loguru import Logger


class MotionClockUtil:
    """Tracks elapsed milliseconds from an external clock source, handling backward clock jumps."""

    __slots__ = ("_anchor", "_clock_sub", "_elapsed_ms_at_anchor", "_last")

    def __init__(self, clock_sub: ClockSub):
        self._clock_sub = clock_sub
        self._anchor: int | None = None
        self._last: int | None = None
        self._elapsed_ms_at_anchor: int = 0

    def reset(self):
        """Reset clock state and underlying clock source."""
        self._anchor = self._last = None
        self._elapsed_ms_at_anchor = 0
        self._clock_sub.reset_origin()

    def elapsed_ms(self, log: Logger | None = None) -> int:
        """Return elapsed milliseconds since reset, handling backward clock jumps."""
        now = self._clock_sub.get_clock()

        if self._anchor is None:
            self._anchor = now

        if self._last is not None and now < self._last:
            # Clock jumped backwards (e.g., sim reset) - re-anchor preserving progress
            self._elapsed_ms_at_anchor += self._last - self._anchor
            self._anchor = now
            if log:
                log.warning("Clock jumped back; re-anchoring.")

        self._last = now
        return self._elapsed_ms_at_anchor + (now - self._anchor)


class TimestepUtil:
    """Converts elapsed time to timesteps, handling forward jump at start."""

    __slots__ = ("_clock", "_interval_ms", "_start_timestep", "_timestep")

    def __init__(self, clock: MotionClockUtil, interval_ms: float, start_timestep: int = 0):
        self._clock = clock
        self._interval_ms = interval_ms
        self._start_timestep = start_timestep
        self._timestep = start_timestep

    def reset(self, start_timestep: int | None = None):
        """Reset timestep state and underlying clock."""
        if start_timestep is not None:
            self._start_timestep = start_timestep
        self._timestep = self._start_timestep
        self._clock.reset()

    def get_timestep(self, log: Logger | None = None) -> int:
        """Return current timestep based on elapsed time, handling forward jump at start."""
        elapsed = self._clock.elapsed_ms(log)
        elapsed_steps = int(elapsed // self._interval_ms)

        # Handle clock jump ahead at start - re-anchor if we're at start but clock jumped
        if self._timestep == self._start_timestep and elapsed_steps > 1:
            if log:
                log.warning("Clock jumped ahead at start; re-anchoring.")
            self._clock.reset()
            return self._timestep

        self._timestep = elapsed_steps + self._start_timestep
        return self._timestep

    @property
    def timestep(self) -> int:
        """Current timestep (read-only access without clock update)."""
        return self._timestep


class PinocchioRobot:
    def __init__(self, robot_cfg: RobotConfig, urdf_text: str):
        # create pinocchio robot. Prune any movable joint NOT in the policy's
        # dof_names (e.g. the yam gripper's 4 prismatic finger jaws, which ride
        # along unactuated and are absent from the 29-DOF dof_names) so the model
        # contains exactly the actuated joints — otherwise buildModelFromXML
        # builds the extra DOF as movable joints and the joint-count assert below
        # fires. No-op for URDFs whose movable joints == dof_names (rubber hand).
        xml_text = self._create_xml_from_urdf(urdf_text, keep_joint_names=robot_cfg.dof_names)
        self.robot_model = pin.buildModelFromXML(xml_text, pin.JointModelFreeFlyer())
        self.robot_data = self.robot_model.createData()

        # get joint names in pinocchio robot and real robot
        joint_names_in_real_robot = robot_cfg.dof_names
        joint_names_in_pinocchio_robot = [
            name for name in self.robot_model.names if name not in ["universe", "root_joint"]
        ]
        assert len(joint_names_in_pinocchio_robot) == len(joint_names_in_real_robot), (
            "The number of joints in the pinocchio robot and the real robot are not the same"
        )
        self.real2pinocchio_index = get_index_of_a_in_b(joint_names_in_pinocchio_robot, joint_names_in_real_robot)

        # get ref body frame id in pinocchio robot
        self.ref_body_frame_id = self.robot_model.getFrameId(robot_cfg.motion["body_name_ref"][0])

    def fk_and_get_ref_body_orientation_in_world(self, configuration: np.ndarray) -> np.ndarray:
        # forward kinematics
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)

        # get ref body pose in world
        ref_body_pose_in_world = self.robot_data.oMf[self.ref_body_frame_id]
        quaternion = pin.Quaternion(ref_body_pose_in_world.rotation)  # (4, )

        return np.expand_dims(quaternion.coeffs(), axis=0)  # xyzw, (1, 4)

    @staticmethod
    def _create_xml_from_urdf(urdf_text: str, keep_joint_names: list[str] | None = None) -> str:
        """Strip visuals/collisions from URDF text and return XML text.

        If ``keep_joint_names`` is given, also prune every MOVABLE joint whose
        name is not in that list, together with its child-link subtree. This
        keeps the Pinocchio model at exactly the actuated DOF for assets that
        carry extra unactuated joints (e.g. the yam gripper's 4 prismatic finger
        jaws, which are absent from the 29-name dof_names). Fixed joints are
        always kept — Pinocchio absorbs them and they may anchor tracked frames
        (foot/ref bodies). No-op when every movable joint is already kept.
        """
        root = ElementTree.fromstring(urdf_text)

        def _localname(tag: str) -> str:
            # Handle optional XML namespaces by only checking the suffix after '}'.
            return tag.rsplit("}", maxsplit=1)[-1]

        for parent in root.iter():
            for child in list(parent):
                if _localname(child.tag) in {"visual", "collision"}:
                    parent.remove(child)

        if keep_joint_names is not None:
            keep = set(keep_joint_names)
            _movable = {"revolute", "continuous", "prismatic", "floating", "planar"}
            joints = [c for c in root if _localname(c.tag) == "joint"]
            # link_name -> child <joint> elements (for subtree walking).
            children_of: dict[str, list] = {}
            for j in joints:
                p = j.find("parent")
                if p is not None and p.get("link"):
                    children_of.setdefault(p.get("link"), []).append(j)

            joints_to_drop: set = set()
            links_to_drop: set[str] = set()
            # Seed with movable joints that are not in the keep set.
            stack = [
                j for j in joints
                if _localname(j.tag) == "joint"
                and j.get("type") in _movable
                and j.get("name") not in keep
            ]
            while stack:
                j = stack.pop()
                if id(j) in joints_to_drop:
                    continue
                joints_to_drop.add(id(j))
                child_el = j.find("child")
                child_link = child_el.get("link") if child_el is not None else None
                if child_link and child_link not in links_to_drop:
                    links_to_drop.add(child_link)
                    # Recurse: any joint mounted on the dropped link goes too.
                    stack.extend(children_of.get(child_link, []))

            for el in list(root):
                lname = _localname(el.tag)
                if lname == "joint" and id(el) in joints_to_drop:
                    root.remove(el)
                elif lname == "link" and el.get("name") in links_to_drop:
                    root.remove(el)

        xml_text = ElementTree.tostring(root, encoding="unicode")
        if not xml_text.lstrip().startswith("<?xml"):
            xml_text = '<?xml version="1.0"?>\n' + xml_text
        return xml_text
