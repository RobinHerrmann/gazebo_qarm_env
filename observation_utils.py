'''
Provides functions to convert ROS-msgs to observation.

Author:  Robin Herrmann
Created: 2023.07.12
'''

import numpy as np


def process_obs_msg_joints(obs_msg_joints):
    # Return Joint Observation
    ''' EXAMPLE MESSAGE
    sensor_msgs.msg.JointState(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=5, nanosec=359000000), 
    frame_id=''), name=['base_yaw_joint', 'yaw_bicep_joint', 'bicep_forearm_joint', 'forearm_endeffector_joint', 'a1_joint', 
    'a2_joint', 'b1_joint', 'b2_joint'], position=[1.0289739388547048e-05, 0.004617760644798873, -0.003861755097277708, 
    -1.8566567980826676e-06, -0.02508906103626958, 6.402487609413754e-06, -0.025080209871456028, 6.590287561003549e-06], 
    velocity=[0.0004263163197073576, 4.617768848305346, -3.8617976980591315, -4.346529298625436e-05, -25.090374552606583, 
    0.0064035882328390414, -25.081522038483296, 0.006591082770007972], effort=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    '''
    a1_index = obs_msg_joints.name.index("a1_joint")
    a2_index = obs_msg_joints.name.index("a2_joint")

    a1_joint = obs_msg_joints.position[a1_index]
    a2_joint = obs_msg_joints.position[a2_index]

    obs_joints = np.array([a1_joint, a2_joint])
    joints_complete = {"name": obs_msg_joints.name, "position": obs_msg_joints.position}

    return obs_joints, joints_complete

def process_obs_msg_models(obs_msg_models):
    # Return Model Observation
    ''' EXAMPLE MESSAGE 
    gazebo_msgs.msg.ModelStates(name=['ground_plane', '3DPrinterBed', 'qarm_v1'], pose=[geometry_msgs.msg.Pose(
    position=geometry_msgs.msg.Point(x=0.0, y=0.0, z=0.0), orientation=geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)), 
    geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=0.0, y=0.0, z=0.0), orientation=
    geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)), geometry_msgs.msg.Pose(
    position=geometry_msgs.msg.Point(x=3.9990152457472056e-11, y=4.3480550752250705e-11, z=1.9334776833572497e-10), orientation=
    geometry_msgs.msg.Quaternion(x=6.266436018869742e-10, y=-5.2036288744002036e-11, z=-3.4389267516328044e-12, w=1.0))], 
    twist=[geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), 
    angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0)), geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), 
    angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0)), geometry_msgs.msg.Twist(
    linear=geometry_msgs.msg.Vector3(x=-2.5013844088598174e-09, y=-3.6933229906730466e-10, z=2.4503577411419397e-08), 
    angular=geometry_msgs.msg.Vector3(x=2.214367219649369e-07, y=6.594972034982955e-08, z=1.4586351408079845e-09))])
    '''
    
    TARGET_index = obs_msg_models.name.index("target")

    TARGET_pose = obs_msg_models.pose[TARGET_index]

    TARGET_x = TARGET_pose.position.x
    TARGET_y = TARGET_pose.position.y
    TARGET_z = TARGET_pose.position.z
    
    obs_models = np.array([TARGET_x, TARGET_y, TARGET_z])

    return obs_models


def process_obs_msg_links(obs_msg_links):
    # Return TCP (endeffectorplate) position
    ''' EXAMPLE MESSAGE 
    gazebo_msgs.msg.LinkStates(name=['ground_plane::link', '3DPrinterBed::link_0', 'qarm_v1::base_link', 'qarm_v1::yaw_link', 
    'qarm_v1::bicep_link', 'qarm_v1::forearm_link', 'qarm_v1::endeffector_link', 'qarm_v1::a1_link', 'qarm_v1::a2_link', 'qarm_v1::b1_link', 
    'qarm_v1::b2_link'], pose=[geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=0.0, y=0.0, z=0.0), 
    orientation=geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)), geometry_msgs.msg.Pose(
    position=geometry_msgs.msg.Point(x=0.4, y=0.0, z=0.06), orientation=geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)), 
    geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=-1.1081400663046781e-09, y=1.0893341909900206e-10, z=1.0846847323785905e-09), 
    orientation=geometry_msgs.msg.Quaternion(x=9.456158928973173e-10, y=-3.0445736854713632e-09, z=-1.5489287750465614e-10, w=1.0)), 
    geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=4.7472375100646604e-07, y=-1.0339320074551349e-08, z=9.945596707305882e-10), 
    orientation=geometry_msgs.msg.Quaternion(x=-1.156823751309256e-08, y=-6.605721298103702e-07, z=-3.1403365514882796e-07, 
    w=0.9999999999997325)), geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=3.2375246539719156e-07, y=-9.002530633956734e-09, 
    z=0.14000069576526372), orientation=geometry_msgs.msg.Quaternion(x=-0.70710788628585, y=0.000508865774843909, z=0.0005084204137129143, 
    w=0.7071053102025635)), geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=0.05050805021532076, y=1.233672019357963e-06, 
    z=0.4899294672893587), orientation=geometry_msgs.msg.Quaternion(x=-0.5010178195331628, y=-0.498981928262221, z=-0.49898042874888326, 
    w=0.5010156798978764)), geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=0.2905131197816976, y=1.0792871080227628e-06, 
    z=0.48895207538611235), orientation=geometry_msgs.msg.Quaternion(x=-0.7085447647542878, y=2.772023418522367e-07, z=-0.7056658673420095, 
    w=-2.768316934454278e-06)), geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=0.34101323104243214, y=-0.030754210579180263, 
    z=0.488744278048903), orientation=geometry_msgs.msg.Quaternion(x=-0.5009129623140578, y=-0.4970428555415981, z=0.49905057302920985, 
    w=0.5029742831398305)), geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=0.3404678827528545, y=-0.10075928614507954, 
    z=0.4887378168459896), orientation=geometry_msgs.msg.Quaternion(x=-0.5008964472145363, y=-0.4970594980888507, z=0.49906749962525504, 
    w=0.5029574886064535)), geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=0.34101349687693255, y=0.030755935145553318, 
    z=0.4887440561466093), orientation=geometry_msgs.msg.Quaternion(x=0.5009076555658022, y=-0.49704550795610664, z=-0.4990523629698397, 
    w=0.5029751709880391)), geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=0.34046877269863884, y=0.10076101522266442, 
    z=0.48873709735055065), orientation=geometry_msgs.msg.Quaternion(x=0.5008911134387942, y=-0.4970621715261396, z=-0.4990692819371652, 
    w=0.5029583898718516))], twist=[geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), 
    angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0)), geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), 
    angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0)), geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(
    x=-3.5675314184628433e-09, y=-3.0554320046611908e-09, z=3.954970732399088e-08), angular=geometry_msgs.msg.Vector3(
    x=1.3426723451362383e-06, y=2.470829868487614e-06, z=-1.4029824671404227e-08)), geometry_msgs.msg.Twist(
    linear=geometry_msgs.msg.Vector3(x=7.43718427895896e-06, y=8.066831678995836e-07, z=4.1760648318022695e-08), 
    angular=geometry_msgs.msg.Vector3(x=3.6109100058916434e-06, y=-1.7849689968198133e-05, z=6.713208471321673e-05)), 
    geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=5.586863530887554e-06, y=3.902183889248388e-07, z=0.00015320080421151797), 
    angular=geometry_msgs.msg.Vector3(x=2.45151095877914e-06, y=1.4399635917100793, z=6.190180220551182e-05)), 
    geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.5047998019249199, y=2.6332369044245106e-06, z=-0.0723476634586433), 
    angular=geometry_msgs.msg.Vector3(x=4.332483466288673e-06, y=4.076127373653931, z=5.220951516032043e-05)), 
    geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.5023059965253343, y=1.5178844922875801e-05, z=-1.0507209695329758), 
    angular=geometry_msgs.msg.Vector3(x=-5.3445579068573e-06, y=4.076131521030085, z=5.2289552002054106e-05)), 
    geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.5015939508618091, y=-0.0011032089078777893, z=-1.2570662017480998), 
    angular=geometry_msgs.msg.Vector3(x=-0.015142589168148916, y=4.0766722871162155, z=-7.789758390324378)), 
    geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=-0.0437591386315018, y=0.0015511971323002361, z=-1.254608923723684), 
    angular=geometry_msgs.msg.Vector3(x=-0.015008401933773364, y=4.076667846340504, z=-7.722875748926335)), 
    geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.5015898629795733, y=0.0011388440398649089, z=-1.2570677523203009), 
    angular=geometry_msgs.msg.Vector3(x=0.0151721633229157, y=4.0767271133669505, z=7.78961567036258)), 
    geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=-0.04375520848726812, y=-0.0015106756230856768, z=-1.2546110290620762), 
    angular=geometry_msgs.msg.Vector3(x=0.015035436155575694, y=4.076722394944826, z=7.722706553374825))])
    '''

    TCP_index = obs_msg_links.name.index("qarm_v1::endeffector_link")

    TCP_pose = obs_msg_links.pose[TCP_index]
    TCP_position = TCP_pose.position

    TCP_x = TCP_pose.position.x
    TCP_y = TCP_pose.position.y
    TCP_z = TCP_pose.position.z
    
    obs_links = np.array([TCP_x, TCP_y, TCP_z])

    return obs_links


def process_obs_relative(a, b):
    # Calculate relative coordinates
    return b - a