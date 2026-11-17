"""Run with ROS sourced: python3 -m unittest discover -s scripts/tests -p test_arm_launch.py."""

import importlib.util
import unittest
import xml.etree.ElementTree as ET
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


class ArmStatePublishersHousingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "arm_state_publishers_launch",
            cls.source / "ros2/launch/arm_state_publishers.launch.py",
        )
        cls.launch = importlib.util.module_from_spec(spec)
        with patch("ament_index_python.packages.get_package_share_directory",
                   return_value=str(cls.source)):
            spec.loader.exec_module(cls.launch)

    def test_launch_applies_variant_to_selected_instrument_not_arm_generation(self):
        for generation, instrument, variant, expected in (
            ("Classic", "420006", "clinical", "Si/housing_Si_clinical.obj"),
            ("Si", "400006", "training", "Classic/housing_Classic_body_training.dae"),
            ("Classic", "400006", "clinical", "Classic/housing_Classic_body.dae"),
            ("Si", "420006", "training", "Si/housing_Si.obj"),
        ):
            with self.subTest(generation=generation, variant=variant):
                context = LaunchContext()
                context.launch_configurations.update(
                    arm="PSM1", generation=generation, instrument=instrument,
                    housing_variant=variant,
                )
                with patch("ament_index_python.packages.get_package_share_directory",
                           return_value=str(self.source)), patch.object(
                               self.launch, "get_package_share_directory",
                               return_value=str(self.source)):
                    actions = self.launch.create_robot_description(context)
                for action in actions:
                    action.execute(context)
                temporary = Path(context.launch_configurations["robot_description_file"])
                self.addCleanup(temporary.unlink)
                urdf = ET.fromstring(context.launch_configurations["robot_description"])
                mesh = urdf.find("link[@name='PSM1_housing_link']/visual/geometry/mesh")
                self.assertEqual(
                    mesh.get("filename"),
                    f"package://dvrk_model/meshes/instruments/housing/{expected}",
                )

    def test_invalid_housing_variant_is_rejected(self):
        context = LaunchContext()
        context.launch_configurations.update(
            arm="PSM1", generation="Si", housing_variant="unknown",
        )
        with patch.object(self.launch, "get_package_share_directory",
                          return_value=str(self.source)):
            with self.assertRaisesRegex(RuntimeError, "Invalid housing_variant 'unknown'"):
                self.launch.create_robot_description(context)
