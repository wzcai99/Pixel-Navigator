import numpy as np
def habitat_camera_intrinsic(config):
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width, 'The configuration of the depth camera should be the same as rgb camera.'
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height, 'The configuration of the depth camera should be the same as rgb camera.'
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov, 'The configuration of the depth camera should be the same as rgb camera.'
    width = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width
    height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height
    hfov = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    intrinsic_matrix = np.array([[f,0,xc],
                                 [0,f,zc],
                                 [0,0,1]],np.float32)
    return intrinsic_matrix

def get_pointcloud_from_depth(rgb:np.ndarray,depth:np.ndarray,intrinsic:np.ndarray):
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    filter_z,filter_x = np.where(depth>-1)
    depth_values = depth[filter_z,filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2])*depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z,filter_x]
    point_values = np.stack([pixel_x,pixel_z,-pixel_y],axis=-1)
    return filter_x,filter_z,point_values,color_values

def translate_to_world(points:np.ndarray,position:np.ndarray,rotation:np.ndarray):
    extrinsic = np.eye(4)
    extrinsic[0:3,0:3] = rotation 
    extrinsic[0:3,3] = position
    world_points = np.matmul(extrinsic,np.concatenate((points,np.ones((points.shape[0],1))),axis=-1).T).T
    return world_points[:,0:3]

def project_to_camera(points,intrinsic,position,rotation):
    extrinsic = np.eye(4)
    extrinsic[0:3,0:3] = rotation
    extrinsic[0:3,3] = position
    extrinsic = np.linalg.inv(extrinsic)
    camera_points = np.concatenate((points,np.ones((points.shape[0],1))),axis=-1)
    camera_points = np.matmul(extrinsic,camera_points.T).T[:,0:3]
    depth_values = -camera_points[:,2]
    filter_x = (camera_points[:,0] * intrinsic[0][0] / depth_values + intrinsic[0][2])
    filter_z = (-camera_points[:,1] * intrinsic[1][1] / depth_values - intrinsic[1][2] + intrinsic[1][2]*2 - 1)
    return filter_x,filter_z,depth_values

