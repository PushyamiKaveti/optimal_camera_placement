import utilities
import visualize
from gtsam import Point3, Cal3_S2,PinholeCameraCal3_S2
import numpy as np
import os
import time
import random
import frank_wolfe as fwolf
import methods
import argparse
from scipy.optimize import minimize


if __name__ == '__main__':
    ''' construct the 3D world and the trajectory'''
    ''' Sample all the camera configurations. In sim  I have ~300 configs '''
    ''' The goal is to pick the best N among these placeents.'''
    ''' Run greedy first, get a initial baseline.'''
    ''' Use greedy solution as initial value'''

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='runs experiments for different benchmark \
                                                 algorithms for optimal camera placement\n\n')
    parser.add_argument('-s', '--select_k', help='number of cameras to select', default=2)
    parser.add_argument('-t', '--traj_type', help='Type of trajectory 1:circle, 2:side, 3:forward, 4:random', default=1)
    parser.add_argument('-o', '--output_dir', help='Output dir for output bag file', default='.')


    args = parser.parse_args()

    num_points = 80
    num_poses = 20

    K = Cal3_S2(100.0, 100.0, 0.0, 50.0, 50.0)
    ''' Number of cameras to be selected'''
    select_k = args.select_k

    traj = args.traj_type


    ''' Generate the simulation data for greedy, franke-wolfe and RAND'''
    poses, points, measurements, extr_cand, intrinsics, poses_with_noise, points_with_noise = fwolf.generate_simulation_data(K,
                                                                                                                 traj,
                                                                                                                 num_points,
                                                                                                                 num_poses,
                                                                                                                 True)  # args.traj_type

    ''' build the prior FIM '''

    h_prior = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3))
    h_prior[-6:, -6:] = np.eye(6)
    # h_prior[-num_poses * 6:, -num_poses * 6:] = np.eye(num_poses * 6)
    # h_prior[0: -num_poses * 6, 0: -num_poses * 6:] = np.eye(num_points * 3)
    # h_prior = h_prior # * 1e-3

    best_config_g, best_selection_indices, best_score_g = fwolf.greedy_selection_new(measurements, intrinsics, extr_cand, points,
                                                                               poses, select_k, h_prior,
                                                                               metric=methods.Metric.min_eig)


    inf_mats, debug_nr_facs = fwolf.construct_candidate_inf_mats(measurements, intrinsics, extr_cand, points, poses)
    num_cands = len(extr_cand)

    selection_init = np.zeros(num_cands)
    ''' uncomment these lines if we want to use greedy solution as the initial estimate'''
    for i in best_selection_indices:
        selection_init[i] = 1


    ''' call frankewolf iterations'''
    print("################# Franke -wolfe greedy initialization ############################")
    # selection_scipy, selection_scipy_unr, cost_scipy, cost_scipy_unrounded = fwolf.scipy_minimize(inf_mats, h_prior,
    #                                                                                         selection_init, select_k,
    #                                                                                         num_poses)
    selection_fw, selection_fw_unr, cost_fw, cost_fw_unrounded, num_iters = fwolf.franke_wolfe(inf_mats, h_prior, 1000,
                                                                                         selection_init.flatten(),
                                                                                         select_k, num_poses)
    print("The Score for traj franke_wolfe with solution. rounded: {:.9f}, unrounded: {:.9f} ".format(cost_fw,
                                                                                                      cost_fw_unrounded))
    print("selection: ")
    print(np.argwhere(selection_fw == 1))

    print("################# Franke -wolfe some initialization ############################")
    ''' This is to input some initialization'''
    selection_init = np.zeros(num_cands)
    selection_init[0] = 1
    selection_init[1] = 1
    #selection_init[2] = 1


    selection_fw, selection_fw_unr, cost_fw, cost_fw_unrounded, num_iters = fwolf.franke_wolfe(inf_mats, h_prior, 1000,
                                                                                               selection_init.flatten(),
                                                                                               select_k, num_poses)
    print("The Score for traj franke_wolfe with solution. rounded: {:.9f}, unrounded: {:.9f} ".format(cost_fw,
                                                                                                      cost_fw_unrounded))
    print("selection: ")
    print(np.argwhere(selection_fw == 1))

    print("################# Franke -wolfe equal weights ############################")
    ''' uncomment these lines if we want to give equal weight to all configurations as initial point'''
    selection_init = np.ones(num_cands)
    selection_init = selection_init*select_k/num_cands

    selection_fw, selection_fw_unr, cost_fw, cost_fw_unrounded, num_iters = fwolf.franke_wolfe(inf_mats, h_prior, 1000,
                                                                                               selection_init.flatten(),
                                                                                               select_k, num_poses)
    print("The Score for traj franke_wolfe with solution. rounded: {:.9f}, unrounded: {:.9f} ".format(cost_fw,
                                                                                                      cost_fw_unrounded))
    print("selection: ")
    print(np.argwhere(selection_fw == 1))