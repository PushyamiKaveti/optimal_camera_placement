import utilities
import visualize
import methods
from gtsam import Point3, Cal3_S2,PinholeCameraCal3_S2
import numpy as np

import gtsam
import math
from gtsam import (DoglegOptimizer,
                    GenericProjectionFactorCal3_S2,
                    NonlinearFactorGraph,
                    PriorFactorPoint3, PriorFactorPose3,  Values)
from numpy import linalg as la

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

def build_graph(measurements, poses, points, extrinsics, ind= -1):
    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
    # factor = PriorFactorPose3(X(0), poses[0], pose_noise)
    # graph.push_back(factor)
    # print("prior factor x0")


    # Simulated measurements from each camera pose, adding them to the factor graph
    for i, pose in enumerate(poses):
        for j, point in enumerate(points):
            for k, comp_pose in enumerate(extrinsics):
                # compose pose with comp_pose T_wb * T_bc = T_wc, we get the pose of component camera W.R.T world
                pose_wc = pose.compose(comp_pose)
                camera = PinholeCameraCal3_S2(pose_wc, K)
                if ind != -1 :
                    assert(len(extrinsics) == 1)
                    measurement = measurements[i,ind,j]
                else:
                    measurement = measurements[i, k, j]
                if measurement[0] == 0 and measurement[1] == 0:
                    continue
                measurement_true = camera.project(point)
                #assert(np.sum(measurement_true-measurement) < 2.0)
                factor = GenericProjectionFactorCal3_S2(
                    measurement, measurement_noise, X(i), L(j), K, comp_pose)
                graph.push_back(factor)

    # print(graph.keys())
    # print("Number of factors in graph"+str(graph.nrFactors()))
    ###graph.print('Factor Graph:\n')

    gt_vals = Values()
    # when we dont have enough constraints for poses.
    num_lms = 0
    num_ps = 0
    pose_mask = np.zeros(len(poses))
    points_mask = np.zeros(len(points))
    for k in graph.keyVector():
        sym = gtsam.Symbol(k)
        # print(sym.chr())
        if sym.chr() == ord('x'):
            num_ps = num_ps + 1
            pose_idx = sym.index()
            gt_vals.insert(X(pose_idx), poses[pose_idx])
            # print('x'+str(pose_idx))
            pose_mask[pose_idx] = 1
        if sym.chr() == ord('l'):
            num_lms = num_lms + 1
            lm_idx = sym.index()
            gt_vals.insert(L(lm_idx), points[lm_idx])
            points_mask[lm_idx] = 1
        # print("l" + str(lm_idx))

    # ins_prior = True
    # point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    # for j, point in enumerate(points):
    #     if j not in rm_lm_indices and j in dict.keys():
    #         if ins_prior:
    #             factor = PriorFactorPoint3(L(j), point, point_noise)
    #             # graph.push_back(factor)
    #             ins_prior = False
    #         gt_vals.insert(L(j), point)
    #         # print("l"+str(j))
    #     else:
    #         points_mask[j] = 0

    # print("num of poses in graph: "+ str(num_ps))
    # print("num of lms in graph: " + str(num_lms))
    if num_ps != len(poses):
        # print("not all poses have observed enough landmarks")
        pass
    # print(graph.keys())
    #print("num of factors: " + str(graph.size()))
    print("Number of factors in graph" + str(graph.nrFactors()))
    return graph, gt_vals, pose_mask, points_mask

'''
1. build the factor graph with all possible combinations of camera placements
2. get the schur complement.
'''
def build_hfull(measurements, points, poses, extr_cand,ind = -1):
    num_poses = len(poses)
    num_points = len(points)
    # points_fil, points_mask = methods.check_and_filter_points(poses, points, extr_cand, K, False)
    #graph, gtvals, poses_mask, points_mask = methods.getMLE_multicam(poses, points, extr_cand, K, points_mask)
    graph, gtvals, poses_mask, points_mask = build_graph(measurements, poses, points, extr_cand, ind)
    fim, crlb = methods.compute_CRLB(gtvals, graph)
    # build full info matrix
    h_full = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3))
    num_ps = np.count_nonzero(poses_mask)
    num_lms = np.count_nonzero(points_mask)
    tot_num_points = len(points)
    '''
    Extract all the block diagonal elements corresponding to the poses (bottom right square block)
    into the full hessian
    '''
    pose_inds = []
    fim_idx = 0
    for val in gtvals.keys():
        sym = gtsam.Symbol(val)
        if sym.chr() == ord('x'):
            h_full_pose_idx = sym.index()
            pose_inds.append(h_full_pose_idx)
            h_full_pose_idx_start = tot_num_points * 3 + h_full_pose_idx * 6
            fim_pose_idx_start = num_lms * 3 + fim_idx * 6
            h_full[h_full_pose_idx_start: h_full_pose_idx_start + 6,
            h_full_pose_idx_start: h_full_pose_idx_start + 6] = fim[fim_pose_idx_start: fim_pose_idx_start + 6,
                                                                fim_pose_idx_start: fim_pose_idx_start + 6]
            fim_idx = fim_idx + 1

    fim_idx = 0
    for val in gtvals.keys():
        sym = gtsam.Symbol(val)
        if sym.chr() == ord('l'):
            idx = sym.index()
            h_full[idx * 3: (idx + 1) * 3, idx * 3: (idx + 1) * 3] = fim[fim_idx * 3: (fim_idx + 1) * 3,
                                                                     fim_idx * 3: (fim_idx + 1) * 3]
            p_idx_fim = 0
            for pose_idx in pose_inds:
                h_full_pose_idx_start = tot_num_points * 3 + pose_idx * 6
                fim_pose_idx_start = num_lms * 3 + p_idx_fim * 6
                h_full[idx * 3: (idx + 1) * 3, h_full_pose_idx_start: h_full_pose_idx_start + 6] = fim[fim_idx * 3: (fim_idx + 1) * 3, fim_pose_idx_start: fim_pose_idx_start + 6]
                h_full[h_full_pose_idx_start: h_full_pose_idx_start + 6, idx * 3: (idx + 1) * 3] = fim[fim_pose_idx_start: fim_pose_idx_start + 6, fim_idx * 3: (fim_idx + 1) * 3]
                p_idx_fim = p_idx_fim + 1
            fim_idx = fim_idx + 1
    # if fim.shape == h_full.shape:
    #     print(np.array_equal(fim, h_full))
    assert (utilities.check_symmetric(h_full))
    return h_full

def construct_candidate_inf_mats(measurements, extr_cand, points, poses):
    num_poses = len(poses)
    num_points = len(points)
    inf_mat_size = (num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3)
    inf_mats = np.zeros((0,inf_mat_size[0], inf_mat_size[1] ))
    h_sum  = np.zeros(inf_mat_size)
    i =0
    for j, cand in enumerate(extr_cand):
        h_cam = build_hfull(measurements, points, poses, [cand], ind=j)
        h_sum = h_sum + h_cam
        inf_mats = np.append(inf_mats,h_cam[None] , axis=0)
        i = i + 1
    print("Number of candidates : "+ str(i))
    hfull = build_hfull(measurements, points, poses, extr_cand)
    hdiff = hfull - h_sum
    print(np.allclose(hfull, h_sum))
    return inf_mats


def generate_measurements(points, poses, extrinsics, toplot=False):
    """
    This method takes in the camera configuration candidates, groundtruth poses and landmarks
    and generates noisy measuerments in the cameras.
    """
    dict = {}
    measurements = np.zeros((len(poses), len(extrinsics),len(points), 2))
    # Simulated measurements from each camera pose, adding them to the factor graph
    for i, pose in enumerate(poses):
        for k, comp_pose in enumerate(extrinsics):
            # compose pose with comp_pose T_wb * T_bc = T_wc, we get the pose of component camera W.R.T world
            pose_wc = pose.compose(comp_pose)
            camera = PinholeCameraCal3_S2(pose_wc, K)
            # print(pose)
            # print(comp_pose)
            # print(pose_wc)
            for j, point in enumerate(points):
                try:
                    measurement = camera.project(point)
                    if (measurement[0] > 1 and measurement[0] < (2 * K.px() - 2) and measurement[1] > 1 and measurement[
                        1] < (2 * K.py() - 2)):
                        measurement = measurement + 1.0 * np.random.randn(2)
                        measurements[i,k,j] = measurement
                        if dict.get(j) is None:
                            dict[j] = 1
                        else:
                            dict[j] = dict[j] + 1
                    else:
                        # print("Measurement is out of bounds: ")
                        # print(measurement)
                        pass

                except Exception:
                    pass
                    # print("Exception at Point")
                    # print(point)

    # This is not needed for localizaruion
    rm_indices = []
    # for k, v in dict.items():
    #     if v < 2:
    #         #print("lm index: " + str(k) + ", lm value : " + str(points[k]))
    #         rm_indices.append(k)
    points_mask = np.ones(len(points))
    for i, pt in enumerate(points):
        if (i in rm_indices) or (i not in dict.keys()):
            points_mask[i] = 0

    return measurements

def find_min_eig_pair(inf_mats,selection, H0):
    inds = np.where(selection > 1e-10)[0]
    #print(selection[inds])
    #final_inf_mat = np.sum(inf_mats[inds], axis=0)
    final_inf_mat = np.zeros(inf_mats[0].shape)
    for i in range(len(inds)):
        final_inf_mat = final_inf_mat + selection[inds[i]] * inf_mats[inds[i]]
    # add prior infomat H0
    final_inf_mat = final_inf_mat + H0
    eigvals, eigvecs = la.eigh(final_inf_mat)
    print(eigvals[0:6])
    return eigvals[0], eigvecs[:,0]

def roundsolution(selection,k):
    idx = np.argpartition(selection, -k)[-k:]
    rounded_sol = np.zeros(len(selection))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol

def franke_wolfe(inf_mats,H0, n_iters, selection_init, k):
    selection_cur= selection_init
    for i in range(n_iters):
        #compute the minimum eigen value and eigen vector
        min_eig_val, min_eig_vec = find_min_eig_pair(inf_mats, selection_cur, H0)
        #Compute gradient
        grad = np.zeros(selection_cur.shape)
        for ind in range(selection_cur.shape[0]):
            grad[ind] = min_eig_vec.T @ inf_mats[ind] @ min_eig_vec
        #round the solution and pick top k
        rounded_sol = roundsolution(grad, k)
        # Step size determination - naive method. No line search
        alpha = 2.0 / (i + 2.0)
        selection_cur = selection_cur + alpha * (rounded_sol - selection_cur)

    print("ended the number of maximum iterations")
    final_solution = roundsolution(selection_cur, k)
    return final_solution


if __name__ == '__main__':
    ''' construct the 3D world and the trajectory'''
    ''' Sample all the camera configurations. In sim  I have ~300 configs '''
    ''' The goal is to pick the best N among these placeents.'''
    ''' Run greedy first, get a initial baseline.'''
    ''' Use greedy solution as initial value'''
    # Setup the parameters
    ''' calibration intrinsic params'''
    K = Cal3_S2(100.0, 100.0, 0.0, 50.0, 50.0)
    best_configs = []
    best_configs_fw = []
    ''' Number of cameras to be selected'''
    select_k = 3
    ''' Generate all possible candidates rotations and translations'''
    pose_rots, pose_trans =  utilities.generate_candidate_poses((0, 2 * math.pi), 4, (-math.pi / 2, math.pi / 2), 4) # (0, math.pi / 2), 1
    ''' generate a room world with landmarks on the 4 walls of the room and a circular trajectory '''
    points, poses_circle = utilities.create_room_world(20, 10, K)
    # points = utilities.createPoints()
    # # Create the set of ground-truth poses
    # poses_circle = utilities.makePoses(K)
    ''' Perform greedy selection method using minimum eigen value metric'''
    # best_config_circle, cost_circle, avail_cands = methods.greedy_selection(points, poses_circle, K, pose_rots, pose_trans,select_k,
    #                                                        metric=methods.Metric.min_eig)
    ''' Brute force selection'''
    # best_config_brute_cirle, cost_brute_circle = brute_force_selection_stereo(points, poses_circle, K, num_cands)
    #print("best config circle: ")
   # print(best_config_circle)
    #print("The logdet for circular traj greedy: {:.2f} ".format(cost_circle))
    #visualize.show_camconfigs(best_config_circle)



    extr_cand = []
    for j, trans in enumerate(pose_trans):
        for k, rot in enumerate(pose_rots):
            cam = gtsam.Pose3(gtsam.Rot3(rot), gtsam.Point3(trans[0], trans[1], trans[2]))
            extr_cand.append(cam)
    ''' #Generate noisy measurements for all the candidate cameras'''
    measurements = generate_measurements(points, poses_circle, extr_cand, False)

    ''' Construct factor graph as if we have all the 300 cameras. edges going between the poses and the landmarks'''
    ''' write the infomat as a combination of the selection variables.'''
    inf_mats = construct_candidate_inf_mats(measurements, extr_cand, points, poses_circle)

    num_cands = len(extr_cand)
    # selection_init = 1-avail_cands
    selection_init = np.zeros(num_cands)
    selection_init[0] = 1
    selection_init[1] = 1
    selection_init[2] = 1
    ''' build the prior FIM '''
    num_poses = len(poses_circle)
    num_points = len(points)
    h_prior = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3))
    h_prior[-num_poses * 6:, -num_poses * 6:] = np.eye(num_poses * 6)
    h_prior[0: -num_poses * 6, 0: -num_poses * 6:] = np.eye(num_points * 3)

    ''' call frankewolf iterations'''
    selected_fw = franke_wolfe(inf_mats,h_prior, 10, selection_init.flatten(), select_k)
    for i in range(selected_fw.shape[0]):
        if selected_fw[i] == 1:
            best_configs_fw.append(extr_cand[i])

    ''' visualize the final selection with franke wolfe'''
    visualize.show_camconfig_with_world(best_configs_fw, K, poses_circle, points)