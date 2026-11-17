"""Run with ROS sourced: python3 -m unittest discover -s scripts/tests."""

import math
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import xacro
import yaml

from scripts.verify_instrument_visual import combined_visual_bounds


class InstrumentAssemblyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = Path(__file__).resolve().parents[2]
        shutil.copytree(source / "urdf", self.root / "urdf")
        self.parts = self.root / "urdf/common/instruments"
        self.table_path = self.parts / "instruments.yaml"
        self.table = yaml.safe_load(self.table_path.read_text())
        self.lookup = patch(
            "ament_index_python.packages.get_package_share_directory",
            return_value=str(self.root),
        )
        self.lookup.start()
        self.addCleanup(self.lookup.stop)

    def expand(self, instrument="420006"):
        self.table_path.write_text(yaml.safe_dump(self.table))
        return ET.fromstring(xacro.process_file(
            str(self.root / "urdf/Si/PSM1.urdf.xacro"),
            mappings={"instrument": instrument},
        ).toxml())

    def test_new_parts_are_selected_without_dispatch_changes(self):
        # A new roll and tip require only their files and a YAML mapping.
        roll = (self.parts / "roll/roll_4670.urdf.xacro").read_text()
        (self.parts / "roll/roll_4321.urdf.xacro").write_text(
            roll.replace("4670", "4321").replace("0.4670", "0.4321")
        )
        tip = (self.parts / "tip/tip_006.urdf.xacro").read_text()
        (self.parts / "tip/tip_999.urdf.xacro").write_text(
            tip.replace("tip_006", "tip_999").replace(
                'name="${prefix}jaw_link"', 'name="${prefix}custom_jaw_link"'
            )
        )
        self.table["instruments"]["429999"] = dict(
            self.table["instruments"]["420006"], roll="4321", tip="999"
        )
        urdf = self.expand("429999")
        roll_origin = urdf.find("joint[@name='roll']/origin")
        self.assertEqual(float(roll_origin.get("xyz").split()[2]), 0.4321)
        self.assertIsNotNone(urdf.find("link[@name='PSM1_custom_jaw_link']"))

    def test_snake_urdf_matches_controller_dh_through_bent_poses(self):
        # Build the tool transform independently from the controller's modified
        # DH rows, then compare with the URDF chain at several coupled poses.
        import numpy as np
        from scripts.verify_instrument_dh import load_json
        from scripts.verify_instrument_visual import rotation_matrix

        def transform(rpy=(0, 0, 0), xyz=(0, 0, 0)):
            result = np.eye(4)
            result[:3, :3] = rotation_matrix(rpy)
            result[:3, 3] = xyz
            return result

        source = Path(__file__).resolve().parents[2]
        tool_dir = source.parents[1] / "cisst-saw/sawIntuitiveResearchKit/share/tool"
        if not tool_dir.is_dir():
            self.skipTest("requires sibling sawIntuitiveResearchKit tool configurations")
        for model in ("400141", "420141"):
            config = load_json(tool_dir / f"ROUND_TIP_SCISSORS_{model}.json")
            urdf = self.expand(model)
            by_child = {joint.find("child").get("link"): joint for joint in urdf.findall("joint")}
            chain = []
            child = "PSM1_tool_tip_link"
            while child != "PSM1_adaptor_link":
                joint = by_child[child]
                chain.append(joint)
                child = joint.find("parent").get("link")
            for roll, pitch, yaw in ((0, 0, 0), (.8, .3, -.2), (-1.2, -.35, .35)):
                angles = dict(roll=roll, wrist_pitch=2*pitch, wrist_yaw=2*yaw,
                              wrist_pitch1=pitch, wrist_pitch2=pitch,
                              wrist_yaw1=yaw, wrist_yaw2=yaw)
                expected = np.eye(4)
                for row in config["DH"]["joints"]:
                    # Controller files store pi/2 rounded to 1.5708.
                    alpha = round(row["alpha"] / (math.pi / 2)) * math.pi / 2
                    offset = round(row["offset"] / (math.pi / 2)) * math.pi / 2
                    expected = (expected @ transform((alpha, 0, 0), (row["A"], 0, 0))
                                @ transform((0, 0, angles[row["name"]] + offset), (0, 0, row["D"])))
                expected = expected @ np.array(config["tooltip_offset"])
                actual = np.eye(4)
                for joint in reversed(chain):
                    origin = joint.find("origin")
                    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0]*3
                    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0]*3
                    actual = actual @ transform(rpy, xyz)
                    if joint.get("type") == "revolute":
                        mimic = joint.find("mimic")
                        angle = (angles[mimic.get("joint")] * float(mimic.get("multiplier", "1"))
                                 if mimic is not None else angles[joint.get("name")])
                        actual = actual @ transform((0, 0, angle))
                np.testing.assert_allclose(actual, expected, atol=1e-10)
            # Only six independent arm joints; all four physical wrist
            # joints mimic half of the corresponding published bend angle.
            for name in ("wrist_pitch1", "wrist_yaw1", "wrist_yaw2", "wrist_pitch2"):
                mimic = urdf.find(f"joint[@name='{name}']/mimic")
                self.assertEqual(mimic.get("joint"), name[:-1])
                self.assertEqual(mimic.get("multiplier"), "0.5")
            active = {joint.get("name") for joint in urdf.findall("joint")
                      if joint.get("type") != "fixed" and joint.find("mimic") is None}
            self.assertEqual(active, {"yaw", "pitch", "insertion", "roll",
                                      "wrist_pitch", "wrist_yaw", "jaw"})
            for name in ("jaw_1", "jaw_2"):
                self.assertEqual(urdf.find(f"joint[@name='{name}']/mimic").get("multiplier"), "0.5")
            roll_link = urdf.find("link[@name='PSM1_roll_link']")
            cylinder = roll_link.find("visual/geometry/cylinder")
            self.assertEqual(float(cylinder.get("radius")), .0025)
            roll_z = float(urdf.find("joint[@name='roll']/origin").get("xyz").split()[2])
            shaft_center = float(roll_link.find("visual/origin").get("xyz").split()[2])
            shaft_bottom = roll_z + shaft_center - float(cylinder.get("length")) / 2
            if model == "420141":
                housing_bounds, errors = combined_visual_bounds(source, "housing", urdf.find("link[@name='PSM1_housing_link']"))
                self.assertFalse(errors)
                self.assertAlmostEqual(shaft_bottom, housing_bounds.maximum[2], places=6)
            else:
                self.assertAlmostEqual(shaft_bottom, .041, places=6)

    def test_placeholder_is_explicit(self):
        self.table["instruments"]["420006"]["tip"] = "placeholder"
        urdf = self.expand()
        self.assertIsNotNone(urdf.find("link[@name='PSM1_jaw_link']/visual/geometry/box"))

    def test_housing_variants_preserve_geometry_and_collision(self):
        source = Path(__file__).resolve().parents[2]
        for generation in ("Classic", "Si"):
            for arm in ("PSM1", "PSM2", "PSM3"):
                for instrument, original, alternate, alternate_variant in (
                    ("400006", "Classic/housing_Classic_body.dae",
                     "Classic/housing_Classic_body_training.dae", "training"),
                    ("420006", "Si/housing_Si.obj",
                     "Si/housing_Si_clinical.obj", "clinical"),
                ):
                    entry = str(self.root / f"urdf/{generation}/{arm}.urdf.xacro")
                    baseline = ET.fromstring(xacro.process_file(
                        entry, mappings={"instrument": instrument},
                    ).toxml())
                    mesh_path = f"link[@name='{arm}_housing_link']/visual/geometry/mesh"
                    original_uri = baseline.find(mesh_path).get("filename")
                    self.assertTrue(original_uri.endswith(original))
                    for variant in ("default", "clinical", "training"):
                        with self.subTest(generation=generation, arm=arm,
                                          instrument=instrument, variant=variant):
                            urdf = ET.fromstring(xacro.process_file(
                                entry, mappings={
                                    "instrument": instrument, "housing_variant": variant,
                                },
                            ).toxml())
                            mesh = urdf.find(mesh_path)
                            expected = alternate if variant == alternate_variant else original
                            self.assertEqual(
                                mesh.get("filename"),
                                f"package://dvrk_model/meshes/instruments/housing/{expected}",
                            )
                            self.assertTrue((source / mesh.get("filename").replace(
                                "package://dvrk_model/", "")).is_file())
                            mesh.set("filename", original_uri)
                            self.assertEqual(ET.tostring(urdf), ET.tostring(baseline))

    def test_housing_assets_only_change_cover_materials(self):
        housing = Path(__file__).resolve().parents[2] / "meshes/instruments/housing"
        ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
        original = ET.parse(housing / "Classic/housing_Classic_body.dae")
        training = ET.parse(housing / "Classic/housing_Classic_body_training.dae")
        effect = ".//c:effect[@id='SWMaterial-fx-0']"
        clinical_color = original.find(effect + "//c:diffuse/c:color", ns).text
        training_color = training.find(effect + "//c:diffuse/c:color", ns).text
        self.assertEqual(
            [float(v) for v in training_color.split()],
            [0.748401, 0.014784, 0.001197, 1],
        )
        for component in ("ambient", "diffuse", "specular"):
            path = effect + f"//c:{component}/c:color"
            training.find(path, ns).text = original.find(path, ns).text
        self.assertEqual(ET.tostring(training.getroot()), ET.tostring(original.getroot()))

        obj = (housing / "Si/housing_Si.obj").read_bytes()
        clinical_obj = (housing / "Si/housing_Si_clinical.obj").read_bytes()
        self.assertIn(b"mtllib housing_Si_clinical.mtl", clinical_obj)
        self.assertEqual(
            clinical_obj.replace(b"mtllib housing_Si_clinical.mtl", b"mtllib housing_Si.mtl"),
            obj,
        )
        original_mtl = (housing / "Si/housing_Si.mtl").read_text()
        clinical_mtl = (housing / "Si/housing_Si_clinical.mtl").read_text()
        blue_grey = "Kd " + " ".join(clinical_color.split()[:3])
        red = "Kd " + " ".join(training_color.split()[:3])
        self.assertIn(blue_grey, clinical_mtl.split("newmtl Shape.035\n")[1].split("newmtl")[0])
        self.assertEqual(clinical_mtl.replace(blue_grey, red), original_mtl)

    def test_invalid_housing_variant_fails(self):
        for instrument in ("400006", "420006"):
            with self.subTest(instrument=instrument), self.assertRaises(xacro.XacroException):
                xacro.process_file(
                    str(self.root / "urdf/Si/PSM1.urdf.xacro"),
                    mappings={"instrument": instrument, "housing_variant": "unknown"},
                )

    def test_virtual_instruments_still_omit_housing_for_both_variants(self):
        for arm in ("PSM1", "PSM2", "PSM3"):
            for variant in ("clinical", "training"):
                with self.subTest(arm=arm, variant=variant):
                    urdf = ET.fromstring(xacro.process_file(
                        str(self.root / f"urdf/Virtual/{arm}.urdf.xacro"),
                        mappings={"housing_variant": variant},
                    ).toxml())
                    self.assertIsNone(urdf.find(f"link[@name='{arm}_housing_link']"))

    def test_missing_part_fails_instead_of_using_placeholder(self):
        self.table["instruments"]["420006"]["tip"] = "999"
        with self.assertRaises(xacro.XacroException) as error:
            self.expand()
        self.assertIn("tip_999.urdf.xacro", str(error.exception))

    def test_si_shaft_meets_housing_and_collision_contains_mesh(self):
        # Compare actual asset bounds in adaptor coordinates. The old shaft
        # extended about 90 mm behind the adaptor, through the housing.
        urdf = self.expand()
        source = Path(__file__).resolve().parents[2]
        roll = urdf.find("link[@name='PSM1_roll_link']")
        housing = urdf.find("link[@name='PSM1_housing_link']")
        shaft_bounds, errors = combined_visual_bounds(source, "roll", roll)
        self.assertFalse(errors)
        housing_bounds, errors = combined_visual_bounds(source, "housing", housing)
        self.assertFalse(errors)
        roll_z = float(urdf.find("joint[@name='roll']/origin").get("xyz").split()[2])
        self.assertAlmostEqual(
            roll_z + shaft_bounds.minimum[2], housing_bounds.maximum[2], places=6
        )
        self.assertLess(abs(shaft_bounds.maximum[2]), 0.004)
        collision = roll.find("collision")
        center = float(collision.find("origin").get("xyz").split()[2])
        half_length = float(collision.find("geometry/cylinder").get("length")) / 2
        self.assertLessEqual(center - half_length, shaft_bounds.minimum[2] + 1e-8)
        self.assertGreaterEqual(center + half_length, shaft_bounds.maximum[2] - 1e-8)


if __name__ == "__main__":
    unittest.main()
