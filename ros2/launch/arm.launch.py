import copy
import os
import tempfile
import yaml
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, OpaqueFunction, SetLaunchConfiguration
from launch.actions import SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
from launch import LaunchContext, LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.actions import DeclareLaunchArgument

def psm_instruments():
    yaml_file = os.path.join(
        get_package_share_directory('dvrk_model'),
        'urdf', 'common', 'instruments', 'instruments.yaml')
    with open(yaml_file, 'r') as stream:
        return yaml.safe_load(stream)['instruments']

def supported_psm_instruments_message():
    instruments = psm_instruments()
    return ', '.join(
        '{} ({})'.format(model, instruments[model]['name'])
        for model in sorted(instruments.keys()))


def configure_rviz(context):
    arm = context.launch_configurations['arm']
    generation = context.launch_configurations['generation']
    rviz_generation = 'Si' if generation == 'Virtual' else generation
    rviz_config_file = os.path.join(
        get_package_share_directory('dvrk_model'),
        'rviz', rviz_generation, arm + '.rviz')

    if context.launch_configurations.get('collision_debug', 'false').lower() not in ('true', '1'):
        return [SetLaunchConfiguration('rviz_config_file', rviz_config_file)]

    with open(rviz_config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    displays = config['Visualization Manager']['Displays']
    robot_index = next(
        index for index, display in enumerate(displays)
        if display.get('Class') == 'rviz_default_plugins/RobotModel')
    visual_display = displays[robot_index]
    visual_display['Name'] = 'Visual Geometry'
    visual_display['Alpha'] = 0.35
    visual_display['Visual Enabled'] = True
    visual_display['Collision Enabled'] = False

    collision_display = copy.deepcopy(visual_display)
    collision_display['Name'] = 'Collision Geometry'
    collision_display['Alpha'] = 1.0
    collision_display['Visual Enabled'] = False
    collision_display['Collision Enabled'] = True
    displays.insert(robot_index + 1, collision_display)

    temporary_config = tempfile.NamedTemporaryFile(
        mode='w', suffix='_collision.rviz', prefix='dvrk_model_', delete=False)
    with temporary_config:
        yaml.safe_dump(config, temporary_config, sort_keys=False)
    return [SetLaunchConfiguration('rviz_config_file', temporary_config.name)]

def generate_launch_description():
    arm = LaunchConfiguration('arm')
    generation = LaunchConfiguration('generation')
    instrument = LaunchConfiguration('instrument', default='')
    endoscope = LaunchConfiguration('endoscope', default='')
    simulated = LaunchConfiguration('simulated', default = 'true')
    use_sim_time = LaunchConfiguration('use_sim_time', default = 'false')
    rate = LaunchConfiguration('rate', default = 50.0)  # Hz, default is 10 so we're increasing that a bit.

    # Virtual uses the Virtual URDF, but its simulated PSM system configuration
    # is the same as the Si configuration.
    system_generation = PythonExpression([
        "'Si' if '", generation, "' == 'Virtual' else '", generation, "'"
    ])
    system_json = [
        PathJoinSubstitution([FindPackageShare('dvrk_config'), 'system', '']),
        '/system-', arm, '_', system_generation, '_KIN_SIMULATED.json',
    ]

    # Declare nodes
    dvrk_node = Node(
        package = 'dvrk_robot',
        executable = 'dvrk_system',
        condition = IfCondition(simulated),
        arguments = ['-j', system_json],
        output = 'log',
    )

    publisher_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('dvrk_model'),
                'launch',
                'arm_state_publishers.launch.py')),
        launch_arguments = {
            'arm': arm,
            'generation': generation,
            'instrument': instrument,
            'endoscope': endoscope,
            'use_sim_time': use_sim_time,
            'rate': rate,
            'suj': 'false',
            'show_rcm': LaunchConfiguration('show_rcm')
        }.items()
    )

    rviz_node = Node(
        package = 'rviz2',
        executable = 'rviz2',
        name = 'rviz2',
        arguments = ['-d', LaunchConfiguration('rviz_config_file')],
        output = 'log',
    )

    ld = LaunchDescription([
        SetEnvironmentVariable('RMW_FASTRTPS_USE_SHM', '0'),
        DeclareLaunchArgument('arm'),
        DeclareLaunchArgument(
            'generation',
            choices=['Classic', 'Si', 'Virtual'],
            description='dVRK system generation'
        ),
        DeclareLaunchArgument(
            'instrument',
            default_value='',
            description='PSM instrument model. Supported: {}'.format(
                supported_psm_instruments_message())),
        DeclareLaunchArgument('endoscope', default_value=''),
        DeclareLaunchArgument('simulated', default_value='true'),
        DeclareLaunchArgument(
            'collision_debug',
            default_value='false',
            description='Overlay collision geometry on translucent visual geometry in RViz'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('rate', default_value='50.0'),
        DeclareLaunchArgument('show_rcm', default_value='true'),
        OpaqueFunction(function=configure_rviz),
        dvrk_node,
        publisher_nodes,
        rviz_node,
    ])

    return ld
