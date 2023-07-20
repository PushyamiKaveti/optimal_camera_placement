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

def build_graph(measurements, poses, points, extrinsics, inds=[], rm_ill_posed=False):
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
    dict = {}
    dict_lm_poses = {}

    if len(inds) == 0:
        inds = range(len(extrinsics))
    # Simulated measurements from each camera pose, adding them to the factor graph
    for i, pose in enumerate(poses):
        for j, point in enumerate(points):
            for k in inds:
                comp_pose = extrinsics[k]
                # compose pose with comp_pose T_wb * T_bc = T_wc, we get the pose of component camera W.R.T world
                pose_wc = pose.compose(comp_pose)
                camera = PinholeCameraCal3_S2(pose_wc, K)
                measurement = measurements[i, k, j]
                if measurement[0] == 0 and measurement[1] == 0:
                    continue
                measurement_true = camera.project(point)
                #assert(np.sum(measurement_true-measurement) < 2.0)
                factor = GenericProjectionFactorCal3_S2(
                    measurement, measurement_noise, X(i), L(j), K, comp_pose)
                graph.push_back(factor)
                if dict.get(j) is None:
                    dict[j] = [graph.nrFactors() - 1]
                else:
                    dict[j] = dict[j] + [graph.nrFactors() - 1]

                if dict_lm_poses.get(j) is None:
                    dict_lm_poses[j] = {i}
                else:
                    dict_lm_poses[j].add(i)
                #
                # if dict_lm_facs.get(j) is None:
                #     dict_lm_facs[j] = [graph.nrFactors() - 1]
                # else:
                #     dict_lm_facs[j] = dict_lm_facs[j] + [graph.nrFactors() - 1]

    # print(graph.keys())
    # print("Number of factors in graph"+str(graph.nrFactors()))
    ###graph.print('Factor Graph:\n')
    rm_indices = []
    rm_lm_indices = []
    for k, v in dict_lm_poses.items():
        # print("lm index: " + str(k)+" factors : "+str(v))
        if len(v) < 2:
            # print("lm index: " + str(k) + ", lm value : " + str(points[k]))
            rm_indices = rm_indices + dict[k]
            rm_lm_indices.append(k)
    # print(graph.keys())
    if rm_ill_posed:
        for i in rm_indices:
            graph.remove(i)

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

    if num_ps != len(poses):
        # print("not all poses have observed enough landmarks")
        pass
    # print(graph.keys())
    #print("num of factors: " + str(graph.size()))
    #print("Number of factors in graph" + str(graph.nrFactors()))
    return graph, gt_vals, pose_mask, points_mask

'''
1. build the factor graph with all possible combinations of camera placements
2. get the schur complement.
'''
def build_hfull(measurements, points, poses, extr_cand,ind = []):
    num_poses = len(poses)
    num_points = len(points)

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
    return h_full,  graph, gtvals, poses_mask, points_mask

def construct_candidate_inf_mats(measurements, extr_cand, points, poses):
    num_poses = len(poses)
    num_points = len(points)
    inf_mat_size = (num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3)
    inf_mats = np.zeros((0,inf_mat_size[0], inf_mat_size[1] ))
    h_sum  = np.zeros(inf_mat_size)
    i =0
    debug_num_facs=[]
    for j, cand in enumerate(extr_cand):
        h_cam, graph, gtvals, poses_mask, points_mask = build_hfull(measurements, points, poses, extr_cand, ind=[j])
        h_sum = h_sum + h_cam
        inf_mats = np.append(inf_mats,h_cam[None] , axis=0)
        debug_num_facs.append(graph.nrFactors())
        i = i + 1
    print("Number of candidates : "+ str(i))
    hfull, _, _ , _, _= build_hfull(measurements, points, poses, extr_cand)
    hdiff = hfull - h_sum
    print(np.allclose(hfull, h_sum))
    return inf_mats,debug_num_facs


def greedy_selection_new(measurements, all_cands, points, poses, Nc, metric= methods.Metric.logdet):
    avail_cand = np.ones((len(all_cands), 1))
    # build the prior FIM
    num_poses = len(poses)
    num_points = len(points)
    h_prior = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3))
    h_prior[-num_poses * 6:, -num_poses * 6:] = np.eye(num_poses * 6)
    h_prior[0: -num_poses * 6, 0: -num_poses * 6:] = np.eye(num_points * 3)
    best_selection_indices = []
    best_config = []
    best_score = 0.0
    # For each camera
    for i in range(0, Nc):
        max_inf = 0.0
        selected_cand = 0
        for j, cand in enumerate(all_cands):
            if avail_cand[j, 0] == 1:
                cur_cands = best_config.copy()
                cur_cands.append(cand)
                cur_selection = best_selection_indices.copy()
                cur_selection.append(j)
                h_full,  graph, gtvals, poses_mask, points_mask = build_hfull(measurements, points, poses, all_cands, cur_selection)
                h_full = h_full + h_prior
                least_fim_eig = 0.0

                fim = methods.compute_schur_fim(h_full, len(poses))
                # least_fim_eig = math.log(np.linalg.det(np.eye(fim.shape[0]) + fim))
                if metric == methods.Metric.logdet:
                    sign, least_fim_eig = np.linalg.slogdet(fim)
                    least_fim_eig = sign * least_fim_eig
                    # print(np.linalg.det(fim))
                    # print(least_fim_eig)
                    # print("-------------------------")
                if metric == methods.Metric.min_eig:
                    assert (utilities.check_symmetric(fim))
                    least_fim_eig = np.linalg.eigvalsh(fim)[0]
                # least_fim_eig = compute_logdet(fim)

                if least_fim_eig > max_inf:
                    max_inf = least_fim_eig
                    selected_cand = j
                    # best_graph = graph
                    # best_gtvals = gt_vals
        best_score = max_inf
        best_config.append(all_cands[selected_cand])
        best_selection_indices.append(selected_cand)
        avail_cand[selected_cand] = 0
        print("Best Score till now: " + str(best_score))
        print("Next best Camera is: ")
        print(best_config[-1])
        print("------------------")

    print("Selected candidates are : ")
    print(np.argwhere(avail_cand.flatten() == 0))

    graph, gtvals, poses_mask, points_mask = build_graph(measurements, poses, points, all_cands, best_selection_indices, True)
    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
    for i in range(len(poses_circle)):
        if poses_mask[i] == 1:
            factor = PriorFactorPose3(X(i), poses_circle[i], pose_noise)
            graph.push_back(factor)
            break
    # print("prior factor x0")

    ''' Give some initial values which are noisy'''
    initial_estimate = Values()
    for val in gtvals.keys():
        '''
        If the resl variable is a pose, store the values
        '''
        if gtsam.Symbol(val).chr() == ord('x'):
            pose = gtvals.atPose3(val)
            transformed_pose = pose.retract(0.1 * np.random.randn(6, 1))
            initial_estimate.insert(val, transformed_pose)
        elif gtsam.Symbol(val).chr() == ord('l'):
            point = gtvals.atPoint3(val)
            transformed_point = point + 0.1 * np.random.randn(3)
            initial_estimate.insert(val, transformed_point)

    # Optimize the graph and print results
    params = gtsam.DoglegParams()
    # params.setVerbosity('ERROR')
    optimizer = DoglegOptimizer(graph, initial_estimate, params)
    try:
        result = optimizer.optimize()
    except Exception:
        result = Values()
        pass
    # result.print('Final results:\n')
    rmse = utilities.compute_traj_error(result, poses_circle)
    print("The RMSE of the estimated trajectory with best camera placement: " + str(rmse))

    return best_config, best_selection_indices, best_score, rmse

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

def find_min_eig_pair(inf_mats,selection, H0, num_poses):
    inds = np.where(selection > 1e-10)[0]
    #print(selection[inds])
    #final_inf_mat = np.sum(inf_mats[inds], axis=0)
    final_inf_mat = np.zeros(inf_mats[0].shape)
    for i in range(len(inds)):
        final_inf_mat = final_inf_mat + selection[inds[i]] * inf_mats[inds[i]]
    # add prior infomat H0
    final_inf_mat = final_inf_mat + H0
    H_schur = methods.compute_schur_fim(final_inf_mat,num_poses )
    eigvals, eigvecs = la.eigh(H_schur)
    #print(selection)
    print(eigvals[0:6])
    return eigvals[0], eigvecs[:,0], final_inf_mat

def roundsolution(selection,k):
    idx = np.argpartition(selection, -k)[-k:]
    rounded_sol = np.zeros(len(selection))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol

def franke_wolfe(inf_mats,H0, n_iters, selection_init, k,num_poses):
    selection_cur= selection_init
    for i in range(n_iters):
        #compute the minimum eigen value and eigen vector
        min_eig_val, min_eig_vec, final_inf_mat = find_min_eig_pair(inf_mats, selection_cur, H0, num_poses)
        #Compute gradient
        grad = np.zeros(selection_cur.shape)
        ''' required for gradient of schur'''
        Hxx = final_inf_mat[-num_poses * 6:, -num_poses * 6:]
        Hll = final_inf_mat[0: -num_poses * 6, 0: -num_poses * 6:]
        Hlx = final_inf_mat[0: -num_poses * 6, -num_poses * 6:]
        for ind in range(selection_cur.shape[0]):
            #grad[ind] = min_eig_vec.T @ inf_mats[ind] @ min_eig_vec
            #gradient schur
            Hc = inf_mats[ind]
            Hxx_c = Hc[-num_poses * 6:, -num_poses * 6:]
            Hll_c = Hc[0: -num_poses * 6, 0: -num_poses * 6:]
            Hlx_c = Hc[0: -num_poses * 6, -num_poses * 6:]
            t0 = Hlx.T
            t1 = np.linalg.inv(Hll)
            t2 = t0 @ t1
            grad_schur = Hxx_c - (Hlx_c.T @ t1 @ t0.T  - t2 @ Hll_c @ t1 @ t0.T + t2 @ Hlx_c )
            grad[ind] = min_eig_vec.T @ grad_schur @ min_eig_vec

        #round the solution and pick top k
        rounded_sol = roundsolution(grad, k)
        # Step size determination - naive method. No line search
        alpha = 0.5 / (i + 2.0) #original 2.0 / (i + 2.0). Was playing around with this.
        selection_cur = selection_cur + alpha * (rounded_sol - selection_cur)

    print("ended the number of maximum iterations")
    final_solution = roundsolution(selection_cur, k)
    min_eig_val, _, _ = find_min_eig_pair(inf_mats, selection_cur, H0, num_poses)
    return final_solution, min_eig_val


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
    pose_rots, pose_trans =  utilities.generate_candidate_poses((0, 3/2 * math.pi), 4, (-math.pi / 2, math.pi / 2), 5) #(0, math.pi / 2), 1) #(-math.pi / 2, math.pi / 2), 4) #
    ''' generate a room world with landmarks on the 4 walls of the room and a circular trajectory '''
    points, poses_circle = utilities.create_room_world(20, 10, K)

    # sideward
    height = 5.0
    up = Point3(0, 0, 1)
    position = Point3(0, 0, height)
    target = Point3(0.0, 3.0, height)
    camera = PinholeCameraCal3_S2.Lookat(position, target, up, K)
    rott = camera.pose().rotation().matrix()
    #poses_circle = utilities.create_forward_side_robot_traj(rott, np.array([-5.0, 0.0, 5.0]), 20, False)
    #poses_circle = utilities.create_forward_side_robot_traj(rott, np.array([0.0, -5.0, 5.0]), 20)
    poses_circle = utilities.create_random_robot_traj(rott, np.array([0.0, -5.0, 5.0]), 20)
    visualize.show_trajectories(poses_circle, points, K, 2, "side_traj")


    extr_cand = []
    for j, trans in enumerate(pose_trans):
        for k, rot in enumerate(pose_rots):
            cam = gtsam.Pose3(gtsam.Rot3(rot), gtsam.Point3(trans[0], trans[1], trans[2]))
            extr_cand.append(cam)
    ''' #Generate noisy measurements for all the candidate cameras'''
    measurements = generate_measurements(points, poses_circle, extr_cand, False)
    # for i in range(len(poses_circle)):
    #     for j in range(len(extr_cand)):
    #         print(measurements[i,j,:])


    ''' Perform greedy selection method using minimum eigen value metric'''
    #best_config_circle_g, cost_circle, rmse_g, avail_cands = methods.greedy_selection(points, poses_circle, K, pose_rots, pose_trans,select_k,
    #                                                      metric=methods.Metric.min_eig)
    best_config_g, best_selection_indices, best_score_g, rmse_g = greedy_selection_new(measurements, extr_cand, points, poses_circle, select_k, metric= methods.Metric.min_eig)
    ''' Brute force selection'''
    # best_config_brute_cirle, cost_brute_circle = brute_force_selection_stereo(points, poses_circle, K, num_cands)
    #print("best config circle: ")
    #print(best_config_circle)
    print("The score for traj greedy: {:.2f} ".format(best_score_g))
    #visualize.show_camconfigs(best_config_circle)

    ''' Construct factor graph as if we have all the 300 cameras. edges going between the poses and the landmarks'''
    ''' write the infomat as a combination of the selection variables.'''
    inf_mats, debug_nr_facs = construct_candidate_inf_mats(measurements, extr_cand, points, poses_circle)

    num_cands = len(extr_cand)
    selection_init = np.zeros(num_cands)
    # for i in best_selection_indices:
    #     selection_init[i] = 1
    selection_init[0] = 1
    selection_init[1] = 1
    selection_init[2] = 1
    # selection_init = np.ones(num_cands)
    # selection_init = selection_init*select_k/num_cands
    ''' build the prior FIM '''
    num_poses = len(poses_circle)
    num_points = len(points)
    h_prior = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3))
    h_prior[-num_poses * 6:, -num_poses * 6:] = np.eye(num_poses * 6)
    h_prior[0: -num_poses * 6, 0: -num_poses * 6:] = np.eye(num_points * 3)

    ''' call frankewolf iterations'''
    selected_fw, cost_fw = franke_wolfe(inf_mats,h_prior, 40, selection_init.flatten(), select_k,num_poses)

    inds=[]
    for i in range(selected_fw.shape[0]):
        if selected_fw[i] == 1:
            best_configs_fw.append(extr_cand[i])
            inds.append(i)
    print(inds)
    print("The Score for traj franke_wolfe: {:.2f} ".format(cost_fw))
    '''
       Compute the RMSE for the best camera placement
       '''
    '''build the factor graph with the configuration '''
    graph, gtvals, poses_mask, points_mask = build_graph(measurements, poses_circle, points, extr_cand, inds, True)
    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
    for i in range(len(poses_circle)):
        if poses_mask[i] == 1:
            factor = PriorFactorPose3(X(i), poses_circle[i], pose_noise)
            graph.push_back(factor)
            break
    # print("prior factor x0")

    ''' Give some initial values which are noisy'''
    initial_estimate = Values()
    for val in gtvals.keys():
        '''
        If the resl variable is a pose, store the values
        '''
        if gtsam.Symbol(val).chr() == ord('x'):
            pose = gtvals.atPose3(val)
            transformed_pose = pose.retract(0.1 * np.random.randn(6, 1))
            initial_estimate.insert(val, transformed_pose)
        elif gtsam.Symbol(val).chr() == ord('l'):
            point = gtvals.atPoint3(val)
            transformed_point = point + 0.1 * np.random.randn(3)
            initial_estimate.insert(val, transformed_point)

    # Optimize the graph and print results
    params = gtsam.DoglegParams()
    # params.setVerbosity('ERROR')
    optimizer = DoglegOptimizer(graph, initial_estimate, params)
    try:
        result = optimizer.optimize()
    except Exception:
        result = Values()
        pass
    # result.print('Final results:\n')
    rmse = utilities.compute_traj_error(result, poses_circle)
    print("The RMSE of the estimated trajectory with best camera placement: " + str(rmse))


    ''' visualize the final selection with franke wolfe'''
    visualize.show_camconfig_with_world([best_config_g, best_configs_fw], 3,["greedy", "franke_wolfe"],  K, poses_circle, points)