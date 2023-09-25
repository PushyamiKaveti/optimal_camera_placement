import utilities
import visualize
import methods
from gtsam import Point3, Cal3_S2,PinholeCameraCal3_S2
import numpy as np
import os
import time
import random
import frank_wolfe as fwolf
import argparse


'''
Generate a single dataset via simulation, run the algorithm for different
select_k. log the results

'''
if __name__ == '__main__':
    ''' construct the 3D world and the trajectory'''
    ''' Sample all the camera configurations. In sim  I have ~300 configs '''
    ''' The goal is to pick the best N among these placeents.'''
    ''' Run greedy first, get a initial baseline.'''
    ''' Use greedy solution as initial value'''

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='runs experiments for different benchmark \
                                                 algorithms for optimal camera placement\n\n')

    parser.add_argument('-n', '--num_runs', help='number of runs in the experiment', default=50)
    parser.add_argument('-s', '--select_k', help='number of cameras to select', default=2)
    parser.add_argument('-t', '--traj_type', help='Type of trajectory 1:circle, 2:side, 3:forward, 4:random', default=1)
    parser.add_argument('-o', '--output_dir', help='Output dir for output bag file', default='.')
    parser.add_argument('-c', '--config_file',
                        help='Yaml file which specifies the topic names, frequency of selection and time range',
                        default='config/config.yaml')


    args = parser.parse_args()

    num_points = 25*4
    num_poses =50
    max_select_k = 7
    K = Cal3_S2(100.0, 100.0, 0.0, 50.0, 50.0)
    ''' Number of cameras to be selected'''
    #select_k = args.select_k

    traj = ""
    ''' For a particular type of trajectory'''
    for traj_ind in range(1, 2):
        if traj_ind == 1:
            traj = "circle"
        elif traj_ind == 2:
            traj = "side"
        elif traj_ind == 3:
            traj = "forward"
        elif traj_ind >= 4:
            traj = "random"

        output_file_Path = os.path.join(args.output_dir, traj+"_{}runs".format(args.num_runs))
        if os.path.exists(output_file_Path):
            os.rmdir(output_file_Path)
        os.mkdir(output_file_Path)

        # Setup the parameters
        # best_configs = {i: [] for i in range(2, max_select_k)}
        # best_configs_fw = {i: [] for i in range(2, max_select_k)}

        rmse_greedy_slam = {i: [] for i in range(2, max_select_k)}
        rmse_greedy_loc = {i: [] for i in range(2, max_select_k)}
        rmse_greedy_gt_slam = {i: [] for i in range(2, max_select_k)}
        rmse_greedy_gt_loc = {i: [] for i in range(2, max_select_k)}

        rmse_fw_slam = {i: [] for i in range(2, max_select_k)}
        rmse_fw_loc = {i: [] for i in range(2, max_select_k)}
        rmse_fw_gt_slam = {i: [] for i in range(2, max_select_k)}
        rmse_fw_gt_loc = {i: [] for i in range(2, max_select_k)}

        rmse_scipy_slam = {i: [] for i in range(2, max_select_k)}
        rmse_scipy_loc = {i: [] for i in range(2, max_select_k)}
        rmse_scipy_gt_slam = {i: [] for i in range(2, max_select_k)}
        rmse_scipy_gt_loc = {i: [] for i in range(2, max_select_k)}

        times_g = {i: [] for i in range(2, max_select_k)}
        times_fw = {i: [] for i in range(2, max_select_k)}
        times_scipy = {i: [] for i in range(2, max_select_k)}

        iters_fw = {i: [] for i in range(2, max_select_k)}

        greedy_scores = {i: [] for i in range(2, max_select_k)}
        fw_scores = {i: [] for i in range(2, max_select_k)}
        fw_scores_unrounded = {i: [] for i in range(2, max_select_k)}

        scipy_scores = {i: [] for i in range(2, max_select_k)}
        scipy_scores_unrounded = {i: [] for i in range(2, max_select_k)}

        greedy_selected_cands = {i: [] for i in range(2, max_select_k)}
        fw_selected_cands = {i: [] for i in range(2, max_select_k)}
        scipy_selected_cands = {i: [] for i in range(2, max_select_k)}

        fw_solution_unr_list = {i: [] for i in range(2, max_select_k)}
        scipy_solution_unr_list = {i: [] for i in range(2, max_select_k)}


        '''
        EQUAL, RAND and STANDARD
        '''
        equal_scores = {i: [] for i in range(2, max_select_k)}
        stnd_scores = {i: [] for i in range(2, max_select_k)}
        rand_scores = {i: [] for i in range(2, max_select_k)}
        equal_selected_cands = {i: [] for i in range(2, max_select_k)}
        stnd_selected_cands = {i: [] for i in range(2, max_select_k)}
        rand_selected_cands = {i: [] for i in range(2, max_select_k)}

        rmse_equal_slam ={i: [] for i in range(2, max_select_k)}
        rmse_equal_loc  ={i: [] for i in range(2, max_select_k)}
        rmse_stnd_slam = {i: [] for i in range(2, max_select_k)}
        rmse_stnd_loc  = {i: [] for i in range(2, max_select_k)}
        rmse_rand_slam = {i: [] for i in range(2, max_select_k)}
        rmse_rand_loc  = {i: [] for i in range(2, max_select_k)}
        prior_scale = 1e-3
        h_prior = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3))
        h_prior[-6:, -6:] = np.eye(6) * (1/prior_scale)
        # h_prior[-num_poses * 6:, -num_poses * 6:] = np.eye(num_poses * 6)
        # h_prior[0: -num_poses * 6, 0: -num_poses * 6:] = np.eye(num_points * 3)
        # h_prior = h_prior #* 1e-3

        ''' Run a set of experiments'''
        for exp_ind in range(args.num_runs):
            ''' Generate the simulation data for greedy, franke-wolfe and RAND'''
            poses, points, measurements, extr_cand, intrinsics, poses_with_noise, points_with_noise = fwolf.generate_simulation_data(K, traj_ind, num_points, num_poses, False ) #args.traj_type

            ''' write the data to log'''
            utilities.write_data_logs(os.path.join(output_file_Path, "data_log_" + traj + "_{}.txt".format(exp_ind)),
                                      poses, points, extr_cand, measurements, poses_with_noise,
                                      points_with_noise)

            ''' For this same data i.e poses and points, generate extr_cands and measurements for EQUAL and STANDARD'''
            ''' the extr_cands for equal and stnd are a list corresponding to the selectKs as shown = [2,2,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6]'''
            measurements_equal, extr_cand_equal, intrinsics_e = fwolf.generate_meas_extr_EQUAL(poses, points, K, list(range(2,max_select_k)))
            measurements_stnd, extr_cand_stnd, intrinsics_s = fwolf.generate_meas_extr_STANDARD(poses, points, K, list(range(2,max_select_k)))
            ''' write the data to log for EQUAL'''
            utilities.write_data_logs(os.path.join(output_file_Path, "data_log_equal_" + traj + "_{}.txt".format(exp_ind)),
                                      poses, points, extr_cand_equal, measurements_equal, poses_with_noise,
                                      points_with_noise)
            ''' write the data to log for STANDARD'''
            utilities.write_data_logs(os.path.join(output_file_Path, "data_log_standard_" + traj + "_{}.txt".format(exp_ind)),
                                      poses, points, extr_cand_stnd, measurements_stnd, poses_with_noise,
                                      points_with_noise)

            ''' Run optimization on different number of selected candidates'''
            for select_k in range(2, max_select_k):
                '''
                Cases of Equal and Random and standard configurations
                '''
                ###EQUAL case candidates to evaluate
                b_i =0
                for f in range((select_k-1), 1, -1):
                    b_i = b_i + f

                #selection_equal = extr_cand_equal[b_i: b_i + select_k]
                #selection_stnd = extr_cand_stnd[b_i: b_i + select_k]
                selection_equal = list(range(b_i , b_i + select_k))
                selection_stnd = list(range(b_i , b_i + select_k))
                selection_rand = random.sample(range(len(extr_cand)), select_k)


                score_equal = fwolf.compute_info_metric(poses, points, measurements_equal, intrinsics_e, extr_cand_equal, selection_equal, h_prior)
                score_stnd = fwolf.compute_info_metric(poses, points, measurements_stnd,intrinsics_s, extr_cand_stnd, selection_stnd, h_prior)
                score_rand = fwolf.compute_info_metric(poses, points, measurements, intrinsics, extr_cand, selection_rand, h_prior)

                equal_scores[select_k].append(score_equal)
                stnd_scores[select_k].append(score_stnd)
                rand_scores[select_k].append(score_rand)
                equal_selected_cands[select_k].append(selection_equal)
                stnd_selected_cands[select_k].append(selection_stnd)
                rand_selected_cands[select_k].append(selection_rand)

                print("RMSE for EQUAL, STANDARD and RAND")
                rmse_equal = fwolf.compute_rmse(measurements_equal, poses, points, intrinsics_e, extr_cand_equal, selection_equal, poses_with_noise,
                                          points_with_noise, prior_scale)
                rmse_equal_l = fwolf.compute_rmse(measurements_equal, poses, points, intrinsics_e,  extr_cand_equal, selection_equal,
                                            poses_with_noise, points_with_noise, prior_scale, loc=True)

                rmse_equal_slam[select_k].append(rmse_equal)
                rmse_equal_loc[select_k].append(rmse_equal_l)

                rmse_stnd = fwolf.compute_rmse(measurements_stnd, poses, points, intrinsics_s, extr_cand_stnd, selection_stnd,
                                          poses_with_noise, points_with_noise, prior_scale)
                rmse_stnd_l = fwolf.compute_rmse(measurements_stnd, poses, points, intrinsics_s, extr_cand_stnd, selection_stnd,
                                            poses_with_noise, points_with_noise,prior_scale, loc=True)

                rmse_stnd_slam[select_k].append(rmse_stnd)
                rmse_stnd_loc[select_k].append(rmse_stnd_l)

                rmse_rand = fwolf.compute_rmse(measurements, poses, points, intrinsics, extr_cand, selection_rand,
                                          poses_with_noise,points_with_noise, prior_scale)
                rmse_rand_l = fwolf.compute_rmse(measurements, poses, points, intrinsics, extr_cand, selection_rand,
                                            poses_with_noise, points_with_noise, prior_scale, loc=True)

                rmse_rand_slam[select_k].append(rmse_rand)
                rmse_rand_loc[select_k].append(rmse_rand_l)


                #score, rmse, selected candidates

                ''' ###########################################################################################################'''
                best_score_g, best_config_g, selected_inds_g, time_greedy, best_score_fw,best_score_fw_unrounded,\
                best_config_fw, selected_inds_fw, solution_fw_unr, time_fw, num_iters_fw, best_score_scipy, best_score_scipy_unrounded,\
                selected_inds_scipy, solution_scipy_unr, time_scipy = fwolf.run_single_experiment(poses, points, measurements, intrinsics, extr_cand, select_k, h_prior)

                greedy_scores[select_k].append(best_score_g)
                fw_scores[select_k].append(best_score_fw)
                fw_scores_unrounded[select_k].append(best_score_fw_unrounded)
                scipy_scores[select_k].append(best_score_scipy)
                scipy_scores_unrounded[select_k].append(best_score_scipy_unrounded)

                greedy_selected_cands[select_k].append(selected_inds_g)
                fw_selected_cands[select_k].append(selected_inds_fw)
                scipy_selected_cands[select_k].append(selected_inds_scipy)

                fw_solution_unr_list[select_k].append(solution_fw_unr.tolist())
                scipy_solution_unr_list[select_k].append(solution_scipy_unr.tolist())

                times_g[select_k].append(time_greedy)
                times_fw[select_k].append(time_fw)
                times_scipy[select_k].append(time_scipy)
                iters_fw[select_k].append(num_iters_fw)

                '''
                   Compute the RMSEs for the best camera placement
                   '''
                '''---------------------- '''
                print("RMSE for Greedy-------------------------------------")
                rmse_g = fwolf.compute_rmse(measurements, poses, points, intrinsics, extr_cand,selected_inds_g,poses_with_noise, points_with_noise,prior_scale )
                rmse_g_loc = fwolf.compute_rmse(measurements, poses, points, intrinsics, extr_cand, selected_inds_g, poses_with_noise, points_with_noise, prior_scale,loc=True)
                rmse_g_gt = fwolf.compute_rmse(measurements, poses, points, intrinsics, extr_cand, selected_inds_g,  poses,points,prior_scale)
                rmse_g_gt_loc = fwolf.compute_rmse(measurements, poses, points, intrinsics, extr_cand, selected_inds_g, poses,points,prior_scale, loc=True)
                rmse_greedy_slam[select_k].append(rmse_g)
                rmse_greedy_loc[select_k].append(rmse_g_loc)
                rmse_greedy_gt_slam[select_k].append(rmse_g_gt)
                rmse_greedy_gt_loc[select_k].append(rmse_g_gt_loc)

                print("RMSE for Franke-wolfe------------------------------------------------")
                rmse_fw = fwolf.compute_rmse(measurements, poses, points, intrinsics, extr_cand, selected_inds_fw, poses_with_noise, points_with_noise,prior_scale)
                rmse_fw_l = fwolf.compute_rmse(measurements, poses, points, intrinsics, extr_cand, selected_inds_fw, poses_with_noise, points_with_noise,prior_scale, loc=True)
                rmse_fw_gt = fwolf.compute_rmse(measurements, poses, points, intrinsics, extr_cand, selected_inds_fw, poses, points,prior_scale)
                rmse_fw_gt_l = fwolf.compute_rmse(measurements, poses, points, intrinsics, extr_cand, selected_inds_fw, poses, points, prior_scale, loc=True)
                rmse_fw_slam[select_k].append(rmse_fw)
                rmse_fw_loc[select_k].append(rmse_fw_l)
                rmse_fw_gt_slam[select_k].append(rmse_fw_gt)
                rmse_fw_gt_loc[select_k].append(rmse_fw_gt_l)


                visualize.reset()
        for select_k in range(2, max_select_k):
            rmse_errors_g = np.vstack((times_g[select_k], greedy_scores[select_k], rmse_greedy_slam[select_k], rmse_greedy_loc[select_k],rmse_greedy_gt_slam[select_k], rmse_greedy_gt_loc[select_k])).T
            result_log_arr = np.hstack((np.array(greedy_selected_cands[select_k]), rmse_errors_g))

            rmse_errors_fw = np.vstack((iters_fw[select_k], times_fw[select_k], fw_scores[select_k], fw_scores_unrounded[select_k], rmse_fw_slam[select_k], rmse_fw_loc[select_k], rmse_fw_gt_slam[select_k], rmse_fw_gt_loc[select_k])).T
            result_log_arr = np.hstack((result_log_arr,np.array(fw_selected_cands[select_k]), np.array(fw_solution_unr_list[select_k]), rmse_errors_fw))
            '''
            EQUAL, RAND and STANDARD
            '''
            rmse_errors_equal = np.vstack((equal_scores[select_k],rmse_equal_slam[select_k], rmse_equal_loc[select_k])).T
            result_log_arr = np.hstack((result_log_arr, np.array(equal_selected_cands[select_k]), rmse_errors_equal))

            rmse_errors_stnd = np.vstack((stnd_scores[select_k], rmse_stnd_slam[select_k], rmse_stnd_loc[select_k])).T
            result_log_arr = np.hstack((result_log_arr, np.array(stnd_selected_cands[select_k]), rmse_errors_stnd))

            rmse_errors_rand = np.vstack((rand_scores[select_k], rmse_rand_slam[select_k], rmse_rand_loc[select_k])).T
            result_log_arr = np.hstack((result_log_arr, np.array(rand_selected_cands[select_k]), rmse_errors_rand))


            ''' Header : greedy_selection, time_greedy, greedy_score, rmse_slam_g, rmse_loc_g rmse_slam_gt_g, rmse_loc_gt_g, 
                         fw_selection, fw_unrounded, num_iters, time_fw, fw_score, fw_score_unrounded, rmse_slam, rmse_loc, rmse_slam_gt, rmse_loc_gt,
                         eq_selection, eq_score, rmse_eq_slam, rmse_eq_loc,
                         stnd_selection, stnd_score, rmse_stnd_slam, rmse_stnd_loc
                         rand_selection, rand_score, rmse_rand_slam, rmse_rand_loc'''
            np.savetxt(os.path.join(output_file_Path, "results_"+traj+"{}_.txt".format(select_k)), result_log_arr)

    ''' visualize the final selection with franke wolfe'''
   # visualize.show_camconfig_with_world([best_config_g, best_configs_fw], 3,["greedy", "franke_wolfe"],  K, poses, points)


# if __name__ == '__main__':
#     ''' construct the 3D world and the trajectory'''
#     ''' Sample all the camera configurations. In sim  I have ~300 configs '''
#     ''' The goal is to pick the best N among these placeents.'''
#     ''' Run greedy first, get a initial baseline.'''
#     ''' Use greedy solution as initial value'''
#
#     parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
#                                      description='runs experiments for different benchmark \
#                                                  algorithms for optimal camera placement\n\n')
#
#     parser.add_argument('-n', '--num_runs', help='number of runs in the experiment', default=10)
#     parser.add_argument('-s', '--select_k', help='number of cameras to select', default=2)
#     parser.add_argument('-t', '--traj_type', help='Type of trajectory 1:circle, 2:side, 3:forward, 4:random', default=1)
#     parser.add_argument('-o', '--output_dir', help='Output dir for output bag file', default='.')
#     parser.add_argument('-c', '--config_file',
#                         help='Yaml file which specifies the topic names, frequency of selection and time range',
#                         default='config/config.yaml')
#
#     args = parser.parse_args()
#
#     num_points = 20
#     num_poses =20
#
#     K = Cal3_S2(100.0, 100.0, 0.0, 50.0, 50.0)
#     ''' Number of cameras to be selected'''
#     #select_k = args.select_k
#
#     traj=""
#     # if args.traj_type == 1:
#     #     traj = "circle"
#     # elif args.traj_type == 2:
#     #     traj = "side"
#     # elif args.traj_type == 3:
#     #     traj = "forward"
#     # elif args.traj_type == 4:
#     #     traj = "random"
#     for select_k in range(2, 7):
#         output_file_Path = os.path.join(args.output_dir, "{}cams_{}runs".format(select_k, args.num_runs))
#         if os.path.exists(output_file_Path):
#             os.rmdir(output_file_Path)
#         os.mkdir(output_file_Path)
#         # Setup the parameters
#         best_configs = []
#         best_configs_fw = []
#
#         rmse_greedy_slam = []
#         rmse_greedy_loc = []
#         rmse_greedy_gt_slam = []
#         rmse_greedy_gt_loc = []
#
#         rmse_fw_slam = []
#         rmse_fw_loc = []
#         rmse_fw_gt_slam = []
#         rmse_fw_gt_loc = []
#
#         rmse_scipy_slam=[]
#         rmse_scipy_loc=[]
#         rmse_scipy_gt_slam=[]
#         rmse_scipy_gt_loc=[]
#
#
#         times_g = []
#         times_fw = []
#         times_scipy = []
#
#         greedy_scores = []
#         fw_scores = []
#         fw_scores_unrounded = []
#
#         scipy_scores = []
#         scipy_scores_unrounded = []
#
#         greedy_selected_cands = []
#         fw_selected_cands = []
#         scipy_selected_cands = []
#
#         fw_solution_unr_list=[]
#         scipy_solution_unr_list = []
#
#         for traj_ind in range(4, 5):
#             if traj_ind == 1:
#                 traj = "circle"
#             elif traj_ind == 2:
#                 traj = "side"
#             elif traj_ind == 3:
#                 traj = "forward"
#             elif traj_ind >= 4:
#                 traj = "random"
#             for exp_ind in range(args.num_runs):
#                 ''' Generate the simulation data'''
#                 poses, points, measurements, extr_cand, poses_with_noise, points_with_noise = generate_simulation_data(K, traj_ind, num_points, num_poses ) #args.traj_type
#                 ''' write the data to log'''
#                 utilities.write_data_logs(os.path.join(output_file_Path, "data_log_"+traj+"_{}cams_{}.txt".format(select_k, exp_ind)), poses, points, extr_cand, measurements, poses_with_noise,
#                                           points_with_noise)
#
#                 best_score_g, best_config_g, selected_inds_g, time_greedy, best_score_fw,best_score_fw_unrounded,\
#                 best_config_fw, selected_inds_fw, solution_fw_unr, time_fw, best_score_scipy, best_score_scipy_unrounded,\
#                 selected_inds_scipy, solution_scipy_unr, time_scipy = run_single_experiment(poses, points, measurements, extr_cand, select_k)
#
#                 greedy_scores.append(best_score_g)
#                 fw_scores.append(best_score_fw)
#                 fw_scores_unrounded.append(best_score_fw_unrounded)
#                 scipy_scores.append(best_score_scipy)
#                 scipy_scores_unrounded.append(best_score_scipy_unrounded)
#
#                 greedy_selected_cands.append(selected_inds_g)
#                 fw_selected_cands.append(selected_inds_fw)
#                 scipy_selected_cands.append(selected_inds_scipy)
#
#                 fw_solution_unr_list.append(solution_fw_unr.tolist())
#                 scipy_solution_unr_list.append(solution_scipy_unr.tolist())
#
#                 times_g.append(time_greedy)
#                 times_fw.append(time_fw)
#                 times_scipy.append(time_scipy)
#
#                 '''
#                    Compute the RMSEs for the best camera placement
#                    '''
#                 '''---------------------- '''
#                 print("RMSE for Greedy-------------------------------------")
#                 rmse_g = compute_rmse(measurements, poses, points, extr_cand,selected_inds_g,poses_with_noise, points_with_noise )
#                 rmse_g_loc = compute_rmse(measurements, poses, points, extr_cand, selected_inds_g, poses_with_noise, points_with_noise, loc=True)
#                 rmse_g_gt = compute_rmse(measurements, poses, points, extr_cand, selected_inds_g,  poses,points)
#                 rmse_g_gt_loc = compute_rmse(measurements, poses, points, extr_cand, selected_inds_g, poses,points, loc=True)
#                 rmse_greedy_slam.append(rmse_g)
#                 rmse_greedy_loc.append(rmse_g_loc)
#                 rmse_greedy_gt_slam.append(rmse_g_gt)
#                 rmse_greedy_gt_loc.append(rmse_g_gt_loc)
#
#                 print("RMSE for Franke-wolfe------------------------------------------------")
#                 rmse_fw = compute_rmse(measurements, poses, points, extr_cand, selected_inds_fw, poses_with_noise, points_with_noise)
#                 rmse_fw_l = compute_rmse(measurements, poses, points, extr_cand, selected_inds_fw, poses_with_noise, points_with_noise, loc=True)
#                 rmse_fw_gt = compute_rmse(measurements, poses, points, extr_cand, selected_inds_fw, poses, points)
#                 rmse_fw_gt_l = compute_rmse(measurements, poses, points, extr_cand, selected_inds_fw, poses, points, loc=True)
#                 rmse_fw_slam.append(rmse_fw)
#                 rmse_fw_loc.append(rmse_fw_l)
#                 rmse_fw_gt_slam.append(rmse_fw_gt)
#                 rmse_fw_gt_loc.append(rmse_fw_gt_l)
#
#                 print("RMSE for scipy------------------------------------------------")
#                 rmse_scipy = compute_rmse(measurements, poses, points, extr_cand, selected_inds_scipy, poses_with_noise,
#                                        points_with_noise)
#                 rmse_scipy_l = compute_rmse(measurements, poses, points, extr_cand, selected_inds_scipy, poses_with_noise,
#                                          points_with_noise, loc=True)
#                 rmse_scipy_gt = compute_rmse(measurements, poses, points, extr_cand, selected_inds_scipy, poses, points)
#                 rmse_scipy_gt_l = compute_rmse(measurements, poses, points, extr_cand, selected_inds_scipy, poses, points,
#                                             loc=True)
#                 rmse_scipy_slam.append(rmse_scipy)
#                 rmse_scipy_loc.append(rmse_scipy_l)
#                 rmse_scipy_gt_slam.append(rmse_scipy_gt)
#                 rmse_scipy_gt_loc.append(rmse_scipy_gt_l)
#
#                 visualize.reset()
#
#                 # '''build the factor graph with the configuration franke wolfe'''
#                 # graph, gtvals, poses_mask, points_mask = build_graph(measurements, poses, points, extr_cand, selected_inds_fw, True)
#                 # # Add a prior on pose x1. This indirectly specifies where the origin is.
#                 # # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
#                 # pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
#                 # for i in range(len(poses)):
#                 #     if poses_mask[i] == 1:
#                 #         factor = PriorFactorPose3(X(i), poses[i], pose_noise)
#                 #         graph.push_back(factor)
#                 #         break
#                 # for i in range(len(points)):
#                 #     if points_mask[i] == 1:
#                 #         point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
#                 #         factor = PriorFactorPoint3(L(i), points[i], point_noise)
#                 #         graph.push_back(factor)
#                 #         break
#                 # # print("prior factor x0")
#                 #
#                 # ''' Give some initial values which are noisy'''
#                 # initial_estimate = Values()
#                 # n_poses_graph=0
#                 # n_points_graph=0
#                 # for val in gtvals.keys():
#                 #     '''
#                 #     If the resl variable is a pose, store the values
#                 #     '''
#                 #     if gtsam.Symbol(val).chr() == ord('x'):
#                 #         pose_ind = gtsam.Symbol(val).index()
#                 #         transformed_pose = poses_with_noise[pose_ind]
#                 #         # pose = gtvals.atPose3(val)
#                 #         # transformed_pose = pose.retract(0.1 * np.random.randn(6, 1))
#                 #         initial_estimate.insert(val, transformed_pose)
#                 #         n_poses_graph = n_poses_graph+1
#                 #     elif gtsam.Symbol(val).chr() == ord('l'):
#                 #         point_ind = gtsam.Symbol(val).index()
#                 #         transformed_point = points_with_noise[point_ind]
#                 #         # point = gtvals.atPoint3(val)
#                 #         # transformed_point = point + 0.1 * np.random.randn(3)
#                 #         initial_estimate.insert(val, transformed_point)
#                 #         n_points_graph = n_points_graph + 1
#                 # print(poses_mask)
#                 # print("num poses : "+str(n_poses_graph))
#                 # print("num points : " + str(n_points_graph))
#                 # # Optimize the graph and print results
#                 # params1 = gtsam.DoglegParams()
#                 # optimizer1 = DoglegOptimizer(graph, initial_estimate, params1)
#                 # try:
#                 #     result1 = optimizer1.optimize()
#                 # except Exception:
#                 #     result1 = Values()
#                 #     pass
#                 #
#                 # # result.print('Final results:\n')
#                 # rmse1 = utilities.compute_traj_error(result1, poses, initial_estimate)
#                 # print("The RMSE of the estimated trajectory with best camera placement dogleg: " + str(rmse1))
#                 #
#                 # params = gtsam.LevenbergMarquardtParams()
#                 # # params.setVerbosity('ERROR')
#                 # optimizer = LevenbergMarquardtOptimizer(graph, initial_estimate, params)
#                 # try:
#                 #     result = optimizer.optimize()
#                 # except Exception:
#                 #     result = Values()
#                 #     pass
#                 #
#                 # # result.print('Final results:\n')
#                 # rmse = utilities.compute_traj_error(result, poses,initial_estimate)
#                 # print("The RMSE of the estimated trajectory with best camera placement: " + str(rmse))
#             rmse_errors_g = np.vstack((times_g, greedy_scores, rmse_greedy_slam, rmse_greedy_loc,rmse_greedy_gt_slam, rmse_greedy_gt_loc)).T
#             result_log_arr = np.hstack((np.array(greedy_selected_cands), rmse_errors_g))
#
#             rmse_errors_fw = np.vstack((times_fw, fw_scores, fw_scores_unrounded, rmse_fw_slam, rmse_fw_loc, rmse_fw_gt_slam, rmse_fw_gt_loc)).T
#             result_log_arr = np.hstack((result_log_arr,np.array(fw_selected_cands), np.array(fw_solution_unr_list), rmse_errors_fw))
#
#             rmse_errors_scipy = np.vstack((times_scipy, scipy_scores, scipy_scores_unrounded, rmse_scipy_slam, rmse_scipy_loc,
#                                         rmse_scipy_gt_slam, rmse_scipy_gt_loc)).T
#             result_log_arr = np.hstack((result_log_arr, np.array(scipy_selected_cands), np.array(scipy_solution_unr_list), rmse_errors_scipy))
#
#             ''' Header : greedy_selection, time_greedy, greedy_score, rmse_slam_g, rmse_loc_g rmse_slam_gt_g, rmse_loc_gt_g,
#                          fw_selection, time_fw, fw_score, fw_score_unrounded, rmse_slam, rmse_loc, rmse_slam_gt, rmse_loc_gt  '''
#             np.savetxt(os.path.join(output_file_Path, "rmse_errors_"+traj+"{}_.txt".format(select_k)), result_log_arr)
#
#     ''' visualize the final selection with franke wolfe'''
#    # visualize.show_camconfig_with_world([best_config_g, best_configs_fw], 3,["greedy", "franke_wolfe"],  K, poses, points)