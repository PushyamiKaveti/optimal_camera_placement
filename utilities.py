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


'''
Utilities
'''
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

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
    #plt.ion()
    #fig1, ax1 = helper_functions.initialize_3d_plot(number=1, limits=np.array([[-30, 30], [-30, 30], [-30, 30]]),
    #                                                view=[-30, -90])
    # plot GTSAM poses
    curPose = gtsam.Pose3(gtsam.Rot3(start_rot), gtsam.Point3(start_trans))
    poses = []
    if vel_cmds is not None:
        assert( len(vel_cmds) == num_poses)
    for i in range(num_poses):
        poses.append(curPose)
        #plot.plot_pose3_on_axes(ax1, curPose, axis_length=1, P=None, scale=1)
        if vel_cmds is None:
            delTheta = -1.0 * (math.pi / 180.0 * 20) + np.random.rand() * (math.pi / 180.0 * 40)
            w = np.array([[0, delTheta, 0]]).T
            v = np.array([[0.5*math.sin(delTheta), 0, 0.5*math.cos(delTheta)]]).T
            #v = np.array([[-0.2 + np.random.rand(), 0, -0.2 + np.random.rand()]]).T
            vel = np.vstack((w, v))
        else:
            vel = vel_cmds[i]
        curPose = curPose.compose(gtsam.Pose3.Expmap(vel))
        # ax1.set_xlim3d([-10, 10])
        # ax1.set_ylim3d([-10, 10])
        # ax1.set_zlim3d([-10, 10])
        # fig1.canvas.draw()
        # fig1.canvas.flush_events()
        # time.sleep(0.2)
        # ax1.cla()
    return poses

def create_forward_side_robot_traj(start_rot=np.eye(3), start_trans=np.array([0,0,0]), num_poses=30, forward=True):
    #plt.ion()
    #fig1, ax1 = helper_functions.initialize_3d_plot(number=1, limits=np.array([[-30, 30], [-30, 30], [-30, 30]]),
    #                                                view=[-30, -90])
    # plot GTSAM poses
    curPose = gtsam.Pose3(gtsam.Rot3(start_rot), gtsam.Point3(start_trans))
    poses = []
    for i in range(num_poses):
        poses.append(curPose)
        #plot.plot_pose3_on_axes(ax1, curPose, axis_length=1, P=None, scale=1)

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

def create_room_world(num_points,num_poses, K):

    plt.ion()
    fig1, ax1 = visualize.initialize_3d_plot(number=1, limits=np.array([[-30, 30], [-30, 30], [-30, 30]]),
                                                    view=[-90, -90])

    # helper_functions.plot_3d_points(ax1, pp, None, 'bo', markersize=2)
    wall_width = 20
    wall_height = 10

    # generate random points on xy plane
    #num_points = 25
    x = -1.0* (wall_width/2.0) + np.random.rand(num_points) * wall_width
    y = -1.0*(wall_height/2.0) + np.random.rand(num_points) * wall_height
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

    rotated_points_1 = (R_x @ R_y @ points).T + np.array([-wall_width/2.0, 0, 5.0])
    # helper_functions.plot_3d_points(ax1, points.T, None, 'bo', markersize=2)
    visualize.plot_3d_points(ax1, rotated_points_1, None, 'ro', markersize=2)

    rotated_points_2 = (R_x @ points).T + np.array([0.0, wall_width/2.0, 5.0])
    visualize.plot_3d_points(ax1, rotated_points_2, None, 'ro', markersize=2)

    rotated_points_3 = (R_z_neg @ R_x @ points).T + np.array([wall_width/2.0, 0.0, 5.0])
    visualize.plot_3d_points(ax1, rotated_points_3, None, 'ro', markersize=2)

    rotated_points_4 = (R_x @ points).T + np.array([0.0, -wall_width/2.0, 5.0])
    visualize.plot_3d_points(ax1, rotated_points_4, None, 'ro', markersize=2)

    points = np.vstack((rotated_points_1, rotated_points_2, rotated_points_3, rotated_points_4))
    print(rotated_points_1)
    print(points.shape)
    points_gtsam = [Point3(points[i, 0], points[i, 1], points[i, 2]) for i in range(num_points * 4)]
    print(points_gtsam)

    # plot GTSAM poses
    radius = 5.0
    height = 5.0
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

    # position = gtsam.Point3(radius * np.cos(angles[0]), radius * np.sin(angles[0]), height)
    # target = gtsam.Point3(3 * radius * np.cos(angles[0]), 3 * radius * np.sin(angles[0]), height)
    # camera = gtsam.PinholeCameraCal3_S2.Lookat(position, target, up, K)
    # rott = camera.pose().rotation().matrix()
    # trass= camera.pose().translation()
    #
    # poses = create_random_robot_traj(rott, np.array([-5.0,5.0,5.0]) )
    #cid = fig1.canvas.mpl_connect('key_press_event', plot)
    # sanity check
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
        time.sleep(0.1)
        ax1.cla()
        visualize.plot_3d_points(ax1, rotated_points_1, None, 'ro', markersize=2)
        visualize.plot_3d_points(ax1, rotated_points_2, None, 'ro', markersize=2)
        visualize.plot_3d_points(ax1, rotated_points_3, None, 'ro', markersize=2)
        visualize.plot_3d_points(ax1, rotated_points_4, None, 'ro', markersize=2)
        print(pose)
        print("number of projected measurements for pose : " + str(i) + " is : " + str(num_projected))
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

def generate_candidate_poses(azi_range, azi_num_samples, elev_range, elev_num_samples):
    min_baseline = 0.15
    # Sample the space of position and rotation
    pose_rots = sample_rotations_sphere(azi_range, azi_num_samples, elev_range, elev_num_samples, False)
    #pose_rots = sample_rotations_sphere((0, 2 * math.pi), 4, (-math.pi / 2, math.pi / 2), 4, False)
    pose_trans = [[-min_baseline, 0, min_baseline], [0, 0, min_baseline], [min_baseline, 0, min_baseline],
                  [-min_baseline, 0, 0], [min_baseline, 0, 0],
                  [-min_baseline, 0, -min_baseline], [0, 0, -min_baseline], [min_baseline, 0, -min_baseline]]
    return pose_rots, pose_trans

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


def compute_traj_error(result, poses):
    pose_ind = 0
    lm_ind = 0
    exp_pose_ests = np.zeros((len(poses), 6))
    #exp_lm_ests = np.zeros((len(points), 3))
    for res in result.keys():
        '''
        If the resl variable is a pose, store the values
        '''
        if gtsam.Symbol(res).chr() == ord('x'):
            # print(gtsam.Symbol(res))
            est_rot = result.atPose3(res).rotation().rpy()
            est_trans = result.atPose3(res).translation()
            exp_pose_ests[pose_ind] = np.hstack((est_rot, est_trans))
            pose_ind = pose_ind + 1
        # elif gtsam.Symbol(res).chr() == ord('l'):
        #     # print(gtsam.Symbol(res))
        #     exp_lm_ests[lm_ind] = result.atPoint3(res).reshape((1, 3))
        #     lm_ind = lm_ind + 1

    rmse = 0.0
    for i, p in enumerate(poses):
        trans_gt = p.translation()
        trans_hat = exp_pose_ests[i, 3:6]
        rmse = rmse + np.square(np.linalg.norm(trans_gt - trans_hat))
    rmse = rmse / len(poses)
    rmse = math.sqrt(rmse)
    return rmse