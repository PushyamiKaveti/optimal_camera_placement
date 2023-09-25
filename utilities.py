import numpy as np
from gtsam import Point3, Cal3_S2, Pose3, PinholeCameraCal3_S2
import gtsam
from typing import List
import matplotlib.pyplot as plt
import math
import visualize
from gtsam.utils import plot
import time
from scipy.spatial.transform import Rotation as R

import pandas as pd
import os
import yaml

'''
Utilities
'''

def read_april_tag_data(opti_poses_file, tag_data_file, extr_calib_dir):

    undist_intrinsics = np.genfromtxt(os.path.join(extr_calib_dir, "undistorted_intrinsics.csv"), delimiter=',', dtype=float)
    intrinsics = [Cal3_S2(undist_intrinsics[i, 0],undist_intrinsics[i, 1],0, undist_intrinsics[i, 2],undist_intrinsics[i, 3]) for i in range(undist_intrinsics.shape[0])]


    T_c_i_all = []
    ref_cam_indx = 4
    with open(os.path.join(extr_calib_dir, "camchain.yaml"), 'r') as f:
        dict_calib = yaml.safe_load(f)
        for cam_key in dict_calib.keys():
            camid = cam_key.split("_")[1][-1]
            print("camid:", camid)
            T_c_i = np.array(dict_calib[cam_key])
            T_c_i_all.append(T_c_i)
    #convert all the transforms into cam4
    T_c4_i = T_c_i_all[ref_cam_indx]
    T_c4_c = []
    for T_c_i in T_c_i_all:
        T_c4_c_np = T_c4_i @ np.linalg.inv(T_c_i)
        T_c4_c.append(gtsam.Pose3(T_c4_c_np) )

    #dirs = os.listdir(extr_calib_dir)
    # for dir in dirs:
    #     pathe = os.path.join(extr_calib_dir, dir)
    #     if os.path.isdir(pathe) and "cam" in dir:
    #         calib_file = [f for f in os.listdir(pathe) if "camchain-imucam" in f][0]
    #         with open(os.path.join(pathe, calib_file), 'r') as f:
    #             dict_calib = yaml.load(f)
    #             for cam_key in dict_calib.keys():
    #                 camid = int(dict_calib[cam_key]['rostopic'].split("/")[2][-1])
    #                 T_c_i = np.array(dict_calib[cam_key]['T_cam_imu'])
    #                 extrinsics.append(T_c_i)

    #rough estimate
    r = np.array([[-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, 1]])
    t = [0, 0, 0]
    T_c4_rig = gtsam.Pose3(gtsam.Rot3(r), gtsam.Point3(t[0], t[1], t[2]))

    cols = ['time', 'rx', 'ry', 'rz', 'rw', 'tx', 'ty', 'tz']
    tagcols = ['time', 'camid', 'tagid', 'rx', 'ry', 'rz', 'rw', 'tx', 'ty', 'tz','u1', 'v1', 'u2', 'v2', 'u3', 'v3', 'u4', 'v4',]
    opti_poses_data = pd.read_csv(opti_poses_file,names=cols)
    tag_data =  pd.read_csv(tag_data_file, names=tagcols)

    #go through the rows of opti
    num_poses = opti_poses_data["time"].nunique()
    num_points = tag_data['tagid'].nunique()
    measurements = np.zeros((num_poses, len(T_c4_c), num_points, 2))

    start_index = 0
    poses = []
    points_dict = {}
    measurement_dict = {}
    tag_id_to_pointid = {}
    point_counter=0
    while start_index < len(opti_poses_data):
        start_time = opti_poses_data['time'][start_index]
        indices = opti_poses_data.index[opti_poses_data['time'] == start_time]
        opti_pose_row = opti_poses_data.iloc[start_index]
        # get the optitrack pose data
        T_rig_o = gtsam.Pose3(gtsam.Rot3(opti_pose_row['rw'], opti_pose_row['rx'], opti_pose_row['ry'], opti_pose_row['rz']), gtsam.Point3(opti_pose_row['tx'], opti_pose_row['ty'], opti_pose_row['tz']))
        T_c4_o = T_c4_rig.compose(T_rig_o)
        T_o_c4 = T_c4_o.inverse()
        poses.append(T_o_c4)

        df_tmp = tag_data.iloc[indices]
        for index, row in df_tmp.iterrows():
            #print(row)
            cam = int(row['camid'])
            tagid = int(row['tagid'])
            if tag_id_to_pointid.get(tagid)  is None:
                tag_id_to_pointid[tagid] = [point_counter] # remove list type if only one point
                point_counter = point_counter + 1
                # adding corners also as points
                tag_id_to_pointid[tagid].append(point_counter + 1)
                tag_id_to_pointid[tagid].append(point_counter + 2)
                tag_id_to_pointid[tagid].append(point_counter + 3)
                tag_id_to_pointid[tagid].append(point_counter + 4)
                point_counter = point_counter + 4
                ########################


            # sanity check : average corners = center image coordinates, project the tag center into camera to see if the coordinates matches
            c1 = np.array([row['u1'], row['v1']])
            c2 = np.array([row['u2'], row['v2']])
            c3 = np.array([row['u3'], row['v3']])
            c4 = np.array([row['u4'], row['v4']])
            corner_center = np.mean(np.vstack([c1, c2, c3, c4]), axis =0)
            # add measurements for the rest of the cotrners as well
            measurements[len(poses) - 1, cam, tag_id_to_pointid[tagid]] = c1
            measurements[len(poses) - 1, cam, tag_id_to_pointid[tagid]] = c2
            measurements[len(poses) - 1, cam, tag_id_to_pointid[tagid]] = c3
            measurements[len(poses) - 1, cam, tag_id_to_pointid[tagid]] = c4

            measurements[len(poses)-1, cam ,tag_id_to_pointid[tagid]] = corner_center
            # 3D center of the apriltag w.r.t cam it is seen in
            t_c_tag = gtsam.Point3(row['tx'], row['ty'], row['tz'])
            R_c_tag = gtsam.Rot3(row['rw'], row['rx'], row['ry'], row['rz'])
            T_c_tag = gtsam.Pose3(R_c_tag,t_c_tag)


            #sanity check
            camera = PinholeCameraCal3_S2(gtsam.Pose3(gtsam.Rot3(np.eye(3)), gtsam.Point3(0.0, 0.0, 0.0)), intrinsics[cam])
            measurement_center = camera.project(t_c_tag)
            if np.linalg.norm(measurement_center - corner_center) > 2.0 :
                print("measurement_center: ")
                print(measurement_center)
                print("computed center from corners: ")
                print(corner_center)

            # convert this into world/opti fram
            t_o_tag = T_o_c4.compose(T_c4_c[cam]).transformFrom(t_c_tag)

            #####$ corners
            P1_c = T_c_tag.transformFrom(gtsam.Point3(-5, -5, 0))
            P1_o = T_o_c4.compose(T_c4_c[cam]).transformFrom(P1_c)
            P2_c = T_c_tag.transformFrom(gtsam.Point3(-5, 5, 0))
            P2_o = T_o_c4.compose(T_c4_c[cam]).transformFrom(P2_c)
            P3_c = T_c_tag.transformFrom(gtsam.Point3(5, 5, 0))
            P3_o = T_o_c4.compose(T_c4_c[cam]).transformFrom(P3_c)
            P4_c = T_c_tag.transformFrom(gtsam.Point3(5, -5, 0))
            P4_o = T_o_c4.compose(T_c4_c[cam]).transformFrom(P4_c)

            #uncomment this section and comment the other section if not using all corners as landmarks
            # if points_dict.get(tagid) is not None:
            #     points_dict[tagid].append(t_o_tag)
            # else:
            #     points_dict[tagid] = [t_o_tag]

            if points_dict.get(tagid) is not None:
                points_dict[tagid][0].append(t_o_tag)
                points_dict[tagid][1].append(P1_o)
                points_dict[tagid][2].append(P2_o)
                points_dict[tagid][3].append(P3_o)
                points_dict[tagid][4].append(P4_o)
            else:
                points_dict[tagid] = {0:[t_o_tag]}
                points_dict[tagid]={1: [P1_o]}
                points_dict[tagid]={2: [P2_o]}
                points_dict[tagid]={3: [P3_o]}
                points_dict[tagid]={4: [P4_o]}

        #update start index
        start_index = indices.tolist()[-1] + 1

    print(poses)
    print("---------")
    points = [None] * num_points
    #if corners also there
    points = [None] * num_points
    for key, vals in points_dict.items():
        print("tag id: ", key)
        all_p = np.array(vals[1])
        mean_tag = all_p.mean(axis=0)
        points[tag_id_to_pointid[key]] = mean_tag
        print("mean tag center : ")
        print(mean_tag)
        print("variance: ")
        print(all_p.var(axis=0))

    poses_with_noise = []
    points_with_noise = []
    for p in poses:
        transformed_pose = p.retract(0.1 * np.random.randn(6, 1))
        poses_with_noise.append(transformed_pose)
    for l in points:
        transformed_point = l + 0.1 * np.random.randn(3)
        points_with_noise.append(transformed_point)

    return intrinsics, T_c4_c, poses, points, measurements, poses_with_noise, points_with_noise

def write_data_logs(filename, poses, landmarks,extrinsics, measurements, poses_init, landmarks_init):
    # Simulated measurements from each camera pose, adding them to the factor graph
    with open(filename, 'w') as f:
        #write all the extrinsic/cameras on the rig
        for i, ext in enumerate(extrinsics):
            t = ext.translation()
            q = ext.rotation().quaternion()
            f.write("e "+str(i)+" "+str(t[0])+" "+str(t[1])+" "+str(t[2])+" "+str(q[3])+" "+str(q[0])+" "+str(q[1])+" "+str(q[2])+"\n")
        #write all the poses
        for i, pose in enumerate(poses):
            t = pose.translation()
            q = pose.rotation().quaternion()
            f.write("x "+str(i)+" "+str(t[0])+" "+str(t[1])+" "+str(t[2])+" "+str(q[3])+" "+str(q[0])+" "+str(q[1])+" "+str(q[2])+"\n")
        for i, point in enumerate(landmarks):
            f.write("l " + str(i)+" "+str(point[0]) + " " + str(point[1]) + " " + str(point[2])+ "\n")
        #write all the landmarks
        inds = range(len(extrinsics))
        for i, pose in enumerate(poses):
            for j, point in enumerate(landmarks):
                for k in inds:
                    measurement = measurements[i, k, j]
                    if measurement[0] == 0 and measurement[1] == 0:
                        continue
                    #write the measurement to the file
                    line = "m "+str(i)+" "+str(k)+" "+str(j)+" "+str(measurement[0])+" "+str(measurement[1])+"\n"
                    f.write(line)

        for i, pose in enumerate(poses_init):
            t = pose.translation()
            q = pose.rotation().quaternion()
            f.write("x_i " + str(i) + " " + str(t[0]) + " " + str(t[1]) + " " + str(t[2]) +
                    " " + str(q[3]) + " " + str(q[0]) + " " + str(q[1]) + " " + str(q[2]) + "\n")

        for i, point in enumerate(landmarks_init):
            f.write("l_i " + str(i) + " " + str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + "\n")

    f.close()


def check_symmetric(a, rtol=1e-03, atol=1e-03):
    res = np.allclose(a, a.T, rtol=rtol, atol=atol)
    if not res:
        print("Norm of distace between symmetry", np.linalg.norm(a - a.T))
    return res

def createPoints() -> List[Point3]:
    # Create the set of ground-truth landmarks
    points = [
        Point3(10.0, 10.0, 10.0),
        Point3(-10.0, 10.0, 10.0),
        Point3(-10.0, -10.0, 10.0),
        Point3(10.0, -10.0, 10.0),
        Point3(10.0, 10.0, -10.0),
        Point3(-10.0, 10.0, -10.0),
        Point3(-10.0, -10.0, -10.0),
        Point3(10.0, -10.0, -10.0),
    ]
    return points

def createPoints2(num_points, cubesize):
    x = -(cubesize/2) + np.random.rand(num_points) * cubesize
    y = -(cubesize/2) + np.random.rand(num_points) * cubesize
    z = -(cubesize/2) + np.random.rand(num_points) * cubesize
    points= [ Point3(x[i],y[i],z[i]) for i in range(num_points)]
    return points

def makePoses(K: Cal3_S2) -> List[Pose3]:
    """Generate a set of ground-truth camera poses arranged in a circle about the origin."""
    radius = 40.0
    height = 10.0
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    up = gtsam.Point3(0, 0, 1)
    target = gtsam.Point3(0, 0, 0)
    poses = []
    for theta in angles:
        position = gtsam.Point3(radius * np.cos(theta), radius * np.sin(theta), height)
        camera = gtsam.PinholeCameraCal3_S2.Lookat(position, target, up, K)
        poses.append(camera.pose())
    return poses

def create_corridor_world(num_points, K):
    plt.ion()
    fig1, ax1 = visualize.initialize_3d_plot(number=1, limits=np.array([[-30, 30], [-30, 30], [-30, 30]]),
                                                    view=[-30, -90])

    # helper_functions.plot_3d_points(ax1, pp, None, 'bo', markersize=2)

    # generate random points on xy plane
    #num_points = 40
    x = -10 + np.random.rand(num_points) * 20
    y = -5 + np.random.rand(num_points) * 10
    z = np.zeros((1, num_points))

    # rotate the points
    R_y = np.array([[math.cos(math.pi / 2), 0, math.sin(math.pi / 2)],
                    [0, 1, 0],
                    [-1 * math.sin(math.pi / 2), 0, math.cos(math.pi / 2)]])

    R_z = np.array([[math.cos(math.pi / 2), -1 * math.sin(math.pi / 2), 0.0],
                    [math.sin(math.pi / 2), math.cos(math.pi / 2), 0.0],
                    [0, 0, 1.0]])

    R_z_neg = np.array([[math.cos(math.pi / 2), math.sin(math.pi / 2), 0.0],
                        [-1 * math.sin(math.pi / 2), math.cos(math.pi / 2), 0.0],
                        [0, 0, 1.0]])

    R_x = np.array([[1.0, 0.0, 0.0],
                    [0, math.cos(math.pi / 2), -1 * math.sin(math.pi / 2)],
                    [0, math.sin(math.pi / 2), math.cos(math.pi / 2)]])

    points = np.vstack((x, y, z))
    #left wall
    rotated_points_1 = (R_x @ R_y @ points).T + np.array([-4.5, 0, 5.0])

    # helper_functions.plot_3d_points(ax1, points.T, None, 'bo', markersize=2)
    visualize.plot_3d_points(ax1, rotated_points_1, None, 'ro', markersize=2)

    # #front wall
    num_points = 100
    x = -20 + np.random.rand(num_points) * 40
    y = -5 + np.random.rand(num_points) * 10
    z = np.zeros((1, num_points))
    points = np.vstack((x, y, z))
    rotated_points_2 = (R_x @ points).T + np.array([0.0, 10.5, 5.0])
    rotated_points_2 = rotated_points_2[rotated_points_2[:, 0] < 3.5]
    rotated_points_2 = rotated_points_2[rotated_points_2[:, 0] > -3.5]
    visualize.plot_3d_points(ax1, rotated_points_2, None, 'ro', markersize=2)

    # #front wall
    num_points = 40
    x = -10 + np.random.rand(num_points) * 20
    y = -5 + np.random.rand(num_points) * 10
    z = np.zeros((1, num_points))
    points = np.vstack((x, y, z))
    #right wall
    rotated_points_3 = (R_z_neg @ R_x @ points).T + np.array([4.5, 0.0, 5.0])
    visualize.plot_3d_points(ax1, rotated_points_3, None, 'ro', markersize=2)

    # #back wall
    #rotated_points_4 = (R_x @ points).T + np.array([0.0, -20.5, 5.0])
    #helper_functions.plot_3d_points(ax1, rotated_points_4, None, 'ro', markersize=2)

    points = np.vstack((rotated_points_1, rotated_points_3, rotated_points_2 ))
    points_gtsam = [Point3(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    print(points_gtsam)

    # plot GTSAM poses straight line path
    poses = []
    num_poses = 15
    dist = 15
    vel_cmds = []
    r = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]])
    up = gtsam.Point3(0, 0, 1)
    position = gtsam.Point3(0, -20.0, 5.0)
    target =  gtsam.Point3(0.0, 20.0, 5.0)
    camera = gtsam.PinholeCameraCal3_S2.Lookat(position, target, up, K)

    for i in range(20):
        w = np.array([[0, 0, 0]]).T
        v = np.array([[0, 0, 0.5]]).T
        vel_cmds.append(np.vstack((w, v)))
    poses = create_random_robot_traj(camera.pose().rotation().matrix(), np.array([0,-10.0,5.0]),20, vel_cmds)

    # for i in range(dist):
    #     # r = np.array([[1,0,0],
    #     #           [0,0,1],
    #     #           [0,-1, 0]])
    #
    #     poses.append(gtsam.Pose3(gtsam.Rot3(r), gtsam.Point3(0,-20+2*i,5.0)))
    #     plot.plot_pose3_on_axes(ax1, poses[-1], axis_length=0.1, P=None, scale=1)

    #plt.show()
    # sanity check
    dict={}
    for i, pose in enumerate(poses):
        plot.plot_pose3_on_axes(ax1, pose, axis_length=1, P=None, scale=1)
        camera = PinholeCameraCal3_S2(pose, K)
        num_projected = 0
        for j, point in enumerate(points_gtsam):
            try:
                measurement = camera.project(point) + 1.0 * np.random.randn(2)
                if (measurement[0] > 1 and measurement[0] < 2 * K.px() and measurement[1] > 1 and measurement[
                    1] < 2 * K.py()):
                    num_projected = num_projected + 1
                    plot.plot_point3_on_axes(ax1, point, 'b*')
                    if dict.get(j) is None:
                        dict[j] = 1
                    else:
                        dict[j] = dict[j] + 1
                # else:
                # print("Measurement is out of bounds: ")
                # print(measurement)
            except Exception:
                pass
                # print("Exception at Point")
                # print(point)
        fig1.canvas.draw()
        fig1.canvas.flush_events()
        ax1.cla()
        time.sleep(0.5)
        ax1.set_xlabel("X_axis")
        ax1.set_ylabel("Y_axis")
        ax1.set_zlabel("Z_axis")

        visualize.plot_3d_points(ax1, rotated_points_1, None, 'ro', markersize=2)
        visualize.plot_3d_points(ax1, rotated_points_2, None, 'ro', markersize=2)
        visualize.plot_3d_points(ax1, rotated_points_3, None, 'ro', markersize=2)
        #helper_functions.plot_3d_points(ax1, rotated_points_4, None, 'ro', markersize=2)
        print(pose)
        print("number of projected measurements for pose : " + str(i) + " is : " + str(num_projected))
        plt.show()
    rm_indices=[]
    for k,v in dict.items():
        if v < 3:
            print("lm index: "+str(k)+", lm value : "+ str(points_gtsam[k]))
            rm_indices.append(k)
    final_points_gtsam=[]
    for i,pt in enumerate(points_gtsam):
        if i not in rm_indices:
            final_points_gtsam.append(pt)
    print(len(final_points_gtsam))

    return final_points_gtsam, poses

def create_random_robot_traj(start_rot=np.eye(3), start_trans=np.array([0,0,0]), num_poses=30, vel_cmds=None):

    curPose = gtsam.Pose3(gtsam.Rot3(start_rot), gtsam.Point3(start_trans))
    poses = []
    if vel_cmds is not None:
        assert( len(vel_cmds) == num_poses)
    for i in range(num_poses):
        poses.append(curPose)

        if vel_cmds is None:
            #checkthe bounds
            delTheta_c = -1.0*(math.pi / 180.0 * 30)
            w_l = np.array([[0, delTheta_c , 0]]).T
            v_l = np.array([[4.0*math.sin(delTheta_c), 0, 4.0*math.cos(delTheta_c)]]).T
            vel_l = np.vstack((w_l, v_l))
            checkPose_l = curPose.compose(gtsam.Pose3.Expmap(vel_l))
            bnd_l = checkPose_l.translation()

            delTheta_c = (math.pi / 180.0 * 30)
            w_r = np.array([[0, delTheta_c, 0]]).T
            v_r = np.array([[4.0 * math.sin(delTheta_c), 0, 4.0 * math.cos(delTheta_c)]]).T
            vel_r = np.vstack((w_r, v_r))
            checkPose_r = curPose.compose(gtsam.Pose3.Expmap(vel_r))
            bnd_r = checkPose_r.translation()
            if abs(bnd_l[0]) >= 10 or abs(bnd_l[1]) >= 10:
                vel = np.vstack((w_r, v_r*0.2/4.0))
            elif abs(bnd_r[0] >= 10) or abs(bnd_r[1] >= 10) :
                vel = np.vstack((w_l, v_l*0.2/4.0))
            else:
                ################
                delTheta = -1.0 * (math.pi / 180.0 * 20) + np.random.rand() * (math.pi / 180.0 * 40)
                w = np.array([[0, delTheta, 0]]).T
                v = np.array([[0.3*math.sin(delTheta), 0, 0.3*math.cos(delTheta)]]).T
                #v = np.array([[-0.2 + np.random.rand(), 0, -0.2 + np.random.rand()]]).T
                vel = np.vstack((w, v))
        else:
            vel = vel_cmds[i]
        curPose = curPose.compose(gtsam.Pose3.Expmap(vel))
    return poses

def create_forward_side_robot_traj(start_rot=np.eye(3), start_trans=np.array([0,0,0]), num_poses=30, forward=True):
    # plt.ion()
    # fig1, ax1 = visualize.initialize_3d_plot(number=1, limits=np.array([[-30, 30], [-30, 30], [-30, 30]]),
    #                                                 view=[-30, -90])
    # plot GTSAM poses
    curPose = gtsam.Pose3(gtsam.Rot3(start_rot), gtsam.Point3(start_trans))
    poses = []
    for i in range(num_poses):
        poses.append(curPose)
        # plot.plot_pose3_on_axes(ax1, curPose, axis_length=1, P=None, scale=1)

        w = np.array([[0, 0, 0]]).T
        if forward:
            v = np.array([[0.0, 0, 0.5]]).T
        else:
            v = np.array([[0.5, 0, 0.0]]).T
        #v = np.array([[-0.2 + np.random.rand(), 0, -0.2 + np.random.rand()]]).T
        vel = np.vstack((w, v))

        curPose = curPose.compose(gtsam.Pose3.Expmap(vel))
        # ax1.set_xlim3d([-10, 10])
        # ax1.set_ylim3d([-10, 10])
        # ax1.set_zlim3d([-10, 10])
        # fig1.canvas.draw()
        # fig1.canvas.flush_events()
        # time.sleep(0.2)
        # ax1.cla()
    return poses

def create_room_world(num_points,num_poses, K, add_ground= False, to_plot=False):
    if to_plot:
        plt.ion()
        fig1, ax1 = visualize.initialize_3d_plot(number=1, limits=np.array([[-30, 30], [-30, 30], [-30, 30]]),
                                                        view=[-90, -90])

    # helper_functions.plot_3d_points(ax1, pp, None, 'bo', markersize=2)
    wall_width = 20
    wall_height = 9
    # generate random points on xy plane
    num_points = int(num_points/4)
    if add_ground:
        x_g = -1.0* (wall_width/2.0) + np.random.rand(num_points*2) * wall_width
        y_g = -1.0*(wall_height) + np.random.rand(num_points*2) * (wall_height*2.0)
        z_g = np.zeros((1, num_points*2))

    # rotate the points
    R_y = np.array([[math.cos(math.pi / 2), 0, math.sin(math.pi / 2)],
                    [0, 1, 0],
                    [-1 * math.sin(math.pi / 2), 0, math.cos(math.pi / 2)]])

    R_z = np.array([[math.cos(math.pi / 2), -1 * math.sin(math.pi / 2), 0.0],
                    [math.sin(math.pi / 2), math.cos(math.pi / 2), 0.0],
                    [0, 0, 1.0]])

    R_z_neg = np.array([[math.cos(math.pi / 2), math.sin(math.pi / 2), 0.0],
                        [-1 * math.sin(math.pi / 2), math.cos(math.pi / 2), 0.0],
                        [0, 0, 1.0]])

    R_x = np.array([[1.0, 0.0, 0.0],
                    [0, math.cos(math.pi / 2), -1 * math.sin(math.pi / 2)],
                    [0, math.sin(math.pi / 2), math.cos(math.pi / 2)]])

    x = -1.0 * (wall_width / 2.0) + np.random.rand(num_points) * wall_width
    y = -1.0 * (wall_height / 2.0) + np.random.rand(num_points) * (wall_height)
    z = np.zeros((1, num_points))
    points = np.vstack((x, y, z))

    rotated_points_1 = (R_x @ R_y @ points).T + np.array([-wall_width/2.0, 0, wall_height / 2.0])
    # helper_functions.plot_3d_points(ax1, points.T, None, 'bo', markersize=2)


    x = -1.0 * (wall_width / 2.0) + np.random.rand(num_points) * wall_width
    y = -1.0 * (wall_height / 2.0) + np.random.rand(num_points) * (wall_height)
    z = np.zeros((1, num_points))
    points = np.vstack((x, y, z))
    rotated_points_2 = (R_x @ points).T + np.array([0.0, wall_width/2.0, wall_height / 2.0])


    x = -1.0 * (wall_width / 2.0) + np.random.rand(num_points) * wall_width
    y = -1.0 * (wall_height / 2.0) + np.random.rand(num_points) * (wall_height)
    z = np.zeros((1, num_points))
    points = np.vstack((x, y, z))
    rotated_points_3 = (R_z_neg @ R_x @ points).T + np.array([wall_width/2.0, 0.0, wall_height / 2.0])


    x = -1.0 * (wall_width / 2.0) + np.random.rand(num_points) * wall_width
    y = -1.0 * (wall_height / 2.0) + np.random.rand(num_points) * (wall_height)
    z = np.zeros((1, num_points))
    points = np.vstack((x, y, z))
    rotated_points_4 = (R_x @ points).T + np.array([0.0, -wall_width/2.0, wall_height / 2.0])
    if to_plot:
        visualize.plot_3d_points(ax1, rotated_points_1, None, 'ro', markersize=2)
        visualize.plot_3d_points(ax1, rotated_points_2, None, 'ro', markersize=2)
        visualize.plot_3d_points(ax1, rotated_points_3, None, 'ro', markersize=2)
        visualize.plot_3d_points(ax1, rotated_points_4, None, 'ro', markersize=2)

    num_g_points=0
    if add_ground:
        points_ground = np.vstack((x_g, y_g, z_g)).T
        if to_plot:
            visualize.plot_3d_points(ax1, points_ground, None, 'ro', markersize=2)
        points = np.vstack((rotated_points_1, rotated_points_2, rotated_points_3, rotated_points_4, points_ground))
        num_g_points = num_points*2
    else:
        points = np.vstack((rotated_points_1, rotated_points_2, rotated_points_3, rotated_points_4))
    #print(rotated_points_1)
    #print(points.shape)
    tot_num_points = num_points*4 + num_g_points
    points_gtsam = [Point3(points[i, 0], points[i, 1], points[i, 2]) for i in range(tot_num_points)]
    #print(points_gtsam)

    # plot GTSAM poses
    radius = 5.0
    height = wall_height/2.0
    angles = np.linspace(0, 2 * np.pi, num_poses, endpoint=False)
    up = gtsam.Point3(0, 0, 1)
    target = gtsam.Point3(0, 0, height)
    poses = []

    for theta in angles:
        position = gtsam.Point3(radius * np.cos(theta), radius * np.sin(theta), height)
        #target = gtsam.Point3(3 * radius * np.cos(theta), 3 * radius * np.sin(theta), height)
        camera = gtsam.PinholeCameraCal3_S2.Lookat(position, target, up, K)
        poses.append(camera.pose())
        # plot.plot_pose3_on_axes(ax1, camera.pose(), axis_length=1, P=None, scale=1)

    #cid = fig1.canvas.mpl_connect('key_press_event', plot)
    # sanity check
    if to_plot:
        for i, pose in enumerate(poses):
            plot.plot_pose3_on_axes(ax1, pose, axis_length=1, P=None, scale=1)
            camera = PinholeCameraCal3_S2(pose, K)
            num_projected = 0
            for j, point in enumerate(points_gtsam):
                try:
                    measurement = camera.project(point) + 1.0 * np.random.randn(2)
                    if (measurement[0] > 1 and measurement[0] < 2 * K.px() and measurement[1] > 1 and measurement[
                        1] < 2 * K.py()):
                        num_projected = num_projected + 1
                        plot.plot_point3_on_axes(ax1,point,'b*')
                    # else:
                    # print("Measurement is out of bounds: ")
                    # print(measurement)
                except Exception:
                    pass
                    # print("Exception at Point")
                    # print(point)
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            time.sleep(0.5)
            ax1.cla()
            # visualize.plot_3d_points(ax1, rotated_points_1, None, 'ro', markersize=2)
            # visualize.plot_3d_points(ax1, rotated_points_2, None, 'ro', markersize=2)
            # visualize.plot_3d_points(ax1, rotated_points_3, None, 'ro', markersize=2)
            # visualize.plot_3d_points(ax1, rotated_points_4, None, 'ro', markersize=2)
            visualize.plot_3d_points(ax1, points, None, 'ro', markersize=2)
            #print(pose)
            #print("number of projected measurements for pose : " + str(i) + " is : " + str(num_projected))
        plt.ioff()
    return points_gtsam, poses


'''
A method which reads the file generated by SO3_grid c++
code to sample via hof fibration.
'''
def sample_rotations_hopf(filename):
    global figno
    with open(filename, 'r') as f:
        quat_samples = f.readlines()
    plt.ion()
    fig = plt.figure(figno)
    figno = figno + 1
    axes = fig.add_subplot(111, projection='3d')
    rot_mats = np.zeros((0, 3,3))
    for i, quat in enumerate(quat_samples):
        quat_np = np.fromstring(quat, dtype=float, sep=' ')
        r = R.from_quat(quat_np)
        rmat = r.as_matrix()
        rot_mats = np.append(rot_mats, rmat[None], axis=0)
        c1 = gtsam.Pose3(gtsam.Rot3(rmat), gtsam.Point3(0, 0, 0))
        for j, quat2 in enumerate(quat_samples):

            quat_np2 = np.fromstring(quat2, dtype=float, sep=' ')
            r2 = R.from_quat(quat_np2)
            rmat2 = r2.as_matrix()

            c2 = gtsam.Pose3(gtsam.Rot3(rmat2), gtsam.Point3(0.1, 0, 0))
            plot.plot_pose3_on_axes(axes, c1, 0.1)
            plot.plot_pose3_on_axes(axes, c2, 0.1)
            axes.set_xlim3d([-0.2, 0.2])
            axes.set_ylim3d([-0.2, 0.2])
            axes.set_zlim3d([-0.2, 0.2])
            fig.canvas.draw()
            fig.canvas.flush_events()

            axes.cla()
            #time.sleep(0.2)
    plt.show()
    return rot_mats
'''
A method which samples rotation from a sphere

'''
def sample_rotations_sphere(azi_range, azi_num_samples, elev_range, elev_num_samples, to_plot= False):
    global figno
    if to_plot:
        plt.ion()
        fig = plt.figure(figno)
        figno = figno + 1
        axes = fig.add_subplot(111, projection='3d')
    rot_mats = np.zeros((0, 3, 3))
    #del_azi = 60.0 * math.pi / 180.0
    #del_elev = 30.0 * math.pi / 180.0
    for del_elev in  np.linspace(elev_range[0], elev_range[1], elev_num_samples): #np.linspace(0, math.pi/4, 3):
        for del_azi in np.linspace(azi_range[0], azi_range[1], azi_num_samples):#np.linspace(0, (2*math.pi - math.pi/3), 6):
            R_y = np.array([[math.cos(del_azi), 0, math.sin(del_azi)],
                            [0, 1, 0],
                            [-1 * math.sin(del_azi), 0, math.cos(del_azi)]])

            R_x = np.array([[1.0, 0.0, 0.0],
                            [0, math.cos(del_elev), -1 * math.sin(del_elev)],
                            [0, math.sin(del_elev), math.cos(del_elev)]])
            rotmat =  R_y @ R_x
            rot_mats = np.append(rot_mats, rotmat[None], axis=0)
            c = gtsam.Pose3(gtsam.Rot3(rotmat), gtsam.Point3(0, 0, 0))

            if to_plot:
                plot.plot_pose3_on_axes(axes, c, 0.7)
                # # draw sphere
                # u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
                # x = np.cos(u) * np.sin(v)
                # y = np.sin(u) * np.sin(v)
                # z = np.cos(v)
                # axes.plot_wireframe(x, y, z, color="r")
                axes.quiver(0, 0, 0, rotmat[0,2], rotmat[1,2], rotmat[2,2])
                fig.canvas.draw()
                fig.canvas.flush_events()
                axes.cla()
                time.sleep(0.1)
    return rot_mats

def generate_candidate_poses(azi_range, azi_num_samples, elev_range, elev_num_samples, min_baseline = 0.15):
    # Sample the space of position and rotation
    pose_rots = sample_rotations_sphere(azi_range, azi_num_samples, elev_range, elev_num_samples, False)

    pose_trans = [[-min_baseline, 0, min_baseline], [0, 0, min_baseline], [min_baseline, 0, min_baseline],
                 [-min_baseline, 0, 0], [min_baseline, 0, 0],
                 [-min_baseline, 0, -min_baseline], [0, 0, -min_baseline], [min_baseline, 0, -min_baseline]]
    #pose_trans = [[-2.0 * min_baseline, 0, 0], [-min_baseline, 0, 0], [0, 0, 0], [min_baseline, 0, 0],  [2.0 * min_baseline, 0, 0]]
    return pose_rots, pose_trans

def generate_extr_cands_phy(pose_trans, pose_rots):
    extr_cand=[] # right now this is implemented only for single elevation angle
    for j, trans in enumerate(pose_trans):
        rot_inds = list(range(len(pose_rots)))
        if j ==0: # no right bottom
            del rot_inds[4:6]
        elif j == 1: #no lower half
            del rot_inds[4:9]
        elif j == 2: # no left bottom
            del rot_inds[7:9]
        elif j == 3: # no right half
            del rot_inds[1:6]
        elif j == 4: # no left half
            del rot_inds[7:12]
        elif j == 5: # no upper right
            del rot_inds[1:3]
        elif j == 6 : # no upper half
            del rot_inds[0:3]
            del rot_inds[7:9]
        elif j == 7: # no upper left
            del rot_inds[10:12]
        for k in rot_inds:
            rot = pose_rots[k]
            cam = gtsam.Pose3(gtsam.Rot3(rot), gtsam.Point3(trans[0], trans[1], trans[2]))
            extr_cand.append(cam)
    return extr_cand

def generate_candidate_poses_equal(select_k_list, min_baseline=0.15):
    pose_rots, pose_trans = generate_candidate_poses((0, 330 / 180 * math.pi), 12, (0, math.pi / 2),1)
    extr_cand =[]

    trans = {2:[3, 4], 3:[3,1,4], 4:[3,1,4,6], 5:[1,4,7,5,3], 6:[1,2,7,6,5,0]}
    thetas= {2:[9, 3], 3:[9, 0, 3], 4:[9, 0,3,6], 5:[0,2,5,8,10], 6:[0,2,4,6,8,10]}
    ''' older code'''
    # for two cameras to 7. uncomment below for azimuth of ((0, 3 / 2 * math.pi), 7)
    # trans = {2: [4, 3], 3: [1, 4, 3], 4: [1, 4, 6, 3], 5: [1, 4, 7, 5, 3], 6: [1, 4, 7, 6, 5, 3],
    #          7: [1, 2, 4, 7, 5, 3, 0]}
    # thetas = {2: [90, 270], 3: [0, 90, 270], 4: [0, 90, 180, 270], 5: [0, 72, 144, 216, 288],
    #           6: [0, 60, 120, 180, 240, 300], 7: [0, 45, 90, 135, 225, 270, 315]}
    # for k in select_k_list:
    #     cur_thetas = thetas[k]
    #     cur_trans = trans[k]
    #     for (r, t) in zip(cur_thetas, cur_trans):
    #         R_y = np.array([[math.cos(t), 0, math.sin(t)],
    #                 [0, 1, 0],
    #                 [-1 * math.sin(t), 0, math.cos(t)]])
    #         cam = gtsam.Pose3(gtsam.Rot3(R_y), gtsam.Point3(pose_trans[t][0], pose_trans[t][1], pose_trans[t][2]))
    #         extr_cand.append(cam)
    '''---------------------'''
    for k in select_k_list:
        cur_theta_inds = thetas[k]
        cur_trans_inds = trans[k]
        for (r, t) in zip(cur_theta_inds, cur_trans_inds):
            rot = pose_rots[r]
            cam = gtsam.Pose3(gtsam.Rot3(rot), gtsam.Point3(pose_trans[t][0], pose_trans[t][1], pose_trans[t][2]))
            extr_cand.append(cam)
    return extr_cand

def generate_candidate_poses_stnd(select_k_list, min_baseline=0.15):
    pose_rots, pose_trans = generate_candidate_poses((0, 330 / 180 * math.pi), 12, (0, math.pi / 2), 1)
    extr_cand = []

    trans = {2: [0,2], 3: [3, 1, 4], 4: [3, 0, 2, 4], 5: [3,0,2,4,6], 6: [5,3,0,2,4,7]}
    thetas = {2: [0, 0], 3: [9, 0, 3], 4: [9, 0, 0, 3], 5: [9,0,0,3,6], 6: [9,9,0,0,3,3]}

    for k in select_k_list:
        cur_theta_inds = thetas[k]
        cur_trans_inds = trans[k]
        for (r, t) in zip(cur_theta_inds, cur_trans_inds):
            rot = pose_rots[r]
            cam = gtsam.Pose3(gtsam.Rot3(rot), gtsam.Point3(pose_trans[t][0], pose_trans[t][1], pose_trans[t][2]))
            extr_cand.append(cam)

    # pose_trans = [[-min_baseline, 0, min_baseline], [0, 0, min_baseline], [min_baseline, 0, min_baseline],
    #               [-min_baseline, 0, 0], [min_baseline, 0, 0],
    #               [-min_baseline, 0, -min_baseline], [0, 0, -min_baseline], [min_baseline, 0, -min_baseline]]
    # # for two cameras
    # trans = {2:[0, 2], 3:[1,4,3], 4:[0,2,4, 3], 5:[0, 2, 4, 3, 6], 6: [0,2,4,3,7,5], 7:[0,2,4,7,6,3,5]}
    # thetas= {2:[0, 0], 3:[0, 90, 270], 4:[0, 0,90, 270], 5:[0, 0, 90, 270, 180], 6:[0, 0, 90, 270, 90, 270], 7:[0, 0,90, 90, 180, 270, 270]}
    # for k in select_k_list:
    #     cur_thetas = thetas[k]
    #     cur_trans = trans[k]
    #     for (r, t) in zip(cur_thetas, cur_trans):
    #         R_y = np.array([[math.cos(t), 0, math.sin(t)],
    #                 [0, 1, 0],
    #                 [-1 * math.sin(t), 0, math.cos(t)]])
    #         cam = gtsam.Pose3(gtsam.Rot3(R_y), gtsam.Point3(pose_trans[t][0], pose_trans[t][1], pose_trans[t][2]))
    #         extr_cand.append(cam)
    return extr_cand

'''
Method that takes number of cameras, min baseline between two closest cameras and the angular separation between them
to return the camera matrices
'''
def build_cam_config(num_cams, theta, min_baseline):
    #positions
    tot_bl = (num_cams -1) * min_baseline
    bl_range = tot_bl/2
    x_pos = np.linspace(-1*bl_range, bl_range, num_cams)
    extr=[]
    fig = plt.figure(1)
    axes = fig.add_subplot(111, projection="3d")
    if num_cams == 1:
        rotmat = np.eye(3)
        R_y = np.array([[math.cos(theta), 0, math.sin(theta)],
                        [0, 1, 0],
                        [-1 * math.sin(theta), 0, math.cos(theta)]])
        rotmat_1 = rotmat @ R_y
        c1 = gtsam.Pose3(gtsam.Rot3(rotmat_1), gtsam.Point3(x_pos[0], 0, 0))
        extr.append(c1)
        plot.plot_pose3_on_axes(axes, c1, 0.1)
        #plt.show()
        return extr

    if num_cams % 2 != 0:
        centr_cam = gtsam.Pose3(gtsam.Rot3(np.eye(3)), gtsam.Point3(x_pos[ (num_cams // 2)], 0, 0))
        extr.append(centr_cam)
        plot.plot_pose3_on_axes(axes, centr_cam, 0.1)
    #go from center outwards

    for nc in range(0, num_cams//2):
        rotmat = np.eye(3)
        rot_nag=0.0
        if num_cams % 2 == 0 :
            rot_nag = (nc + 1) * theta - (theta/2)
        else:
            rot_nag = (nc+1) * theta
        R_y = np.array([[math.cos(rot_nag), 0, math.sin(rot_nag)],
                        [0, 1, 0],
                        [-1 * math.sin(rot_nag), 0, math.cos(rot_nag)]])
        R_y_neg = np.array([[math.cos(rot_nag), 0, -1.0 * math.sin(rot_nag)],
                            [0, 1, 0],
                            [math.sin(rot_nag), 0, math.cos(rot_nag)]])

        rotmat_1 = rotmat @ R_y_neg
        print(rotmat_1[:,0].dot(rotmat_1[:,2]))
        c1_ind = (num_cams//2) - (nc +1)
        c1 = gtsam.Pose3(gtsam.Rot3(rotmat_1), gtsam.Point3(x_pos[c1_ind], 0, 0))
        rotmat_2 = rotmat @ R_y
        c2_ind = (num_cams // 2) + (nc + 1)
        if num_cams % 2 == 0:
             c2_ind = c2_ind - 1

        c2 = gtsam.Pose3(gtsam.Rot3(rotmat_2), gtsam.Point3(x_pos[c2_ind], 0, 0))

        extr.append(c1)
        extr.append(c2)
        plot.plot_pose3_on_axes(axes, c1, 0.1)
        plot.plot_pose3_on_axes(axes, c2, 0.1)
    #plt.show()
    return extr


def compute_traj_error(result, poses, initial_estimate=[]):

    lm_ind = 0
    exp_pose_ests = np.zeros((len(poses), 6))
    # exp_lm_ests = np.zeros((len(points), 3))
    rmse = 0.0
    num_poses_err = 0
    for res in result.keys():
        '''
        If the resl variable is a pose, store the values
        '''
        if gtsam.Symbol(res).chr() == ord('x'):
            # print(gtsam.Symbol(res))
            est_rot = result.atPose3(res).rotation().rpy()
            est_trans = result.atPose3(res).translation()
            # print(gtsam.Symbol(res).index())
            # print(poses[gtsam.Symbol(res).index()].translation())
            # print(initial_estimate.atPose3(res).translation())
            # print(est_trans)
            rmse = rmse + np.square(
                np.linalg.norm(est_trans - poses[gtsam.Symbol(res).index()].translation()))
            num_poses_err = num_poses_err + 1
    rmse = rmse / len(poses)
    rmse = math.sqrt(rmse)

    return rmse