"""Run with ROS sourced: python3 -m unittest discover -s scripts/tests -p test_arm_launch.py."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from launch import LaunchContext


class ArmLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "arm_launch", cls.source / "ros2/launch/arm.launch.py"
        )
        cls.launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.launch)

    def create_robot(self, arm, generation, instrument):
        context = LaunchContext()
        context.launch_configurations.update(
            arm=arm, generation=generation, instrument=instrument
        )
        shares = {
            "dvrk_model": str(self.source),
            "dvrk_config": "/config",
        }
        with patch.object(
            self.launch, "get_package_share_directory",
            side_effect=shares.__getitem__,
        ), patch.object(self.launch, "Node") as node:
            result = self.launch.create_simulated_robot(context)
        self.assertEqual(result, [node.return_value])
        return node.call_args.kwargs

    def test_controller_uses_only_supported_json_argument(self):
        cases = [
            ("PSM1", "Si", "420006", "Si"),
            ("PSM2", "Classic", "400006", "Classic"),
            ("PSM1", "Si", "", "Si"),
            ("PSM3", "Classic", "", "Classic"),
            ("PSM1", "Virtual", "420006", "Si"),
            ("PSM1", "Classic", "420006", "Classic"),
            ("ECM", "Si", "", "Si"),
            ("MTML", "Classic", "", "Classic"),
        ]
        for arm, generation, instrument, system_generation in cases:
            with self.subTest(arm=arm, generation=generation, instrument=instrument):
                node = self.create_robot(arm, generation, instrument)
                self.assertEqual(node["package"], "dvrk_robot")
                self.assertEqual(node["executable"], "dvrk_system")
                self.assertEqual(node["arguments"], [
                    "-j",
                    f"/config/system/system-{arm}_{system_generation}_KIN_SIMULATED.json",
                ])

    def test_unsupported_instrument_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported PSM instrument: 999999"):
            self.create_robot("PSM1", "Si", "999999")
