import utilities
from gtsam import Point3, Cal3_S2, PinholeCameraCal3_S2
import numpy as np
import time
import gtsam
from gtsam import (DoglegOptimizer, LevenbergMarquardtOptimizer,
                    GenericProjectionFactorCal3_S2,
                    NonlinearFactorGraph,
                    PriorFactorPoint3, PriorFactorPose3,  Values)
from numpy import linalg as la

from scipy.optimize import minimize, Bounds

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

def compute_CRLB(vals, graph):
    lin_graph = graph.linearize(vals)
    hess = lin_graph.hessian()[0]
    cov = None
    # try:
    #     cov = np.linalg.inv(hess)
    # except Exception:
    #     print("Exception in inverse: info mat is singular")
    #     return hess, None

    return hess, cov

def compute_schur_fim(fim, num_poses):

    Hxx = fim[ -num_poses*6: , -num_poses*6: ]
    Hll = fim[0 : -num_poses*6, 0: -num_poses*6: ]
    Hlx = fim [0: -num_poses*6 , -num_poses*6 : ]

    Hxx_schur = Hxx - Hlx.T @ np.linalg.pinv(Hll) @ Hlx
    return Hxx_schur

def build_graph(measurements, poses, points, intrinsics, extrinsics, inds=[], rm_ill_posed=False):
    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    #pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
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
                camera = PinholeCameraCal3_S2(pose_wc, intrinsics[k])
                measurement = measurements[i, k, j]
                if measurement[0] == 0 and measurement[1] == 0:
                    continue
                # measurement_true = camera.project(point)
                # if (measurement_true - measurement)[0] > 4.0 or (measurement_true - measurement)[1] > 4.0:
                #     print("big error: {:.6f}, {:.6f} ".format((measurement_true - measurement)[0],
                #                                               (measurement_true - measurement)[1]))
                #     cccc = 2.0
                #assert(np.sum(measurement_true-measurement) < 2.0)
                factor = GenericProjectionFactorCal3_S2(
                    measurement, measurement_noise, X(i), L(j), intrinsics[k], comp_pose)
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
    uniq_poses = set()
    for k, v in dict_lm_poses.items():
        # print("lm index: " + str(k)+" poses : "+str(v))
        # print(type(v))
        uniq_poses = uniq_poses.union(v)
        if len(v) < 2:
            # print("lm index: " + str(k) + ", lm value : " + str(points[k]))
            rm_indices = rm_indices + dict[k]
            rm_lm_indices.append(k)
    # print(graph.keys())
    # print("total number of unique poses: {}".format(len(uniq_poses)))
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
def build_hfull(measurements, points, poses, intrinsics, extr_cand,ind = []):
    num_poses = len(poses)
    num_points = len(points)

    graph, gtvals, poses_mask, points_mask = build_graph(measurements, poses, points, intrinsics, extr_cand, ind)

    fim, crlb = compute_CRLB(gtvals, graph)
    # build full info matrix
    h_full = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3), dtype=np.float64)
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

def construct_candidate_inf_mats(measurements, intrinsics, extr_cand, points, poses):
    num_poses = len(poses)
    num_points = len(points)
    inf_mat_size = (num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3)
    inf_mats = np.zeros((0,inf_mat_size[0], inf_mat_size[1] ))
    h_sum  = np.zeros(inf_mat_size)
    i =0
    debug_num_facs=[]
    for j, cand in enumerate(extr_cand):
        h_cam, graph, gtvals, poses_mask, points_mask = build_hfull(measurements, points, poses, intrinsics, extr_cand, ind=[j])
        h_sum = h_sum + h_cam
        inf_mats = np.append(inf_mats,h_cam[None] , axis=0)
        debug_num_facs.append(graph.nrFactors())
        i = i + 1
    print("Number of candidates : "+ str(i))
    # hfull, _, _ , _, _= build_hfull(measurements, points, poses, intrinsics, extr_cand)
    # hdiff = hfull - h_sum
    # print(np.allclose(hfull, h_sum))
    return inf_mats,debug_num_facs

'''
################################################################
Scipy optimization methods
'''
def min_eig_obj_with_jac(x, inf_mats, H0, num_poses):
    # compute the minimum eigen value and eigen vector
    min_eig_val, min_eig_vec, final_inf_mat = find_min_eig_pair(inf_mats, x, H0, num_poses)
    # Compute gradient
    grad = np.zeros(x.shape)
    ''' required for gradient of schur'''
    Hxx = final_inf_mat[-num_poses * 6:, -num_poses * 6:]
    Hll = final_inf_mat[0: -num_poses * 6, 0: -num_poses * 6:]
    Hlx = final_inf_mat[0: -num_poses * 6, -num_poses * 6:]
    for ind in range(x.shape[0]):
        # grad[ind] = min_eig_vec.T @ inf_mats[ind] @ min_eig_vec
        # gradient schur
        Hc = inf_mats[ind]
        Hxx_c = Hc[-num_poses * 6:, -num_poses * 6:]
        Hll_c = Hc[0: -num_poses * 6, 0: -num_poses * 6:]
        Hlx_c = Hc[0: -num_poses * 6, -num_poses * 6:]
        t0 = Hlx.T
        t1 = np.linalg.pinv(Hll)
        t2 = t0 @ t1
        grad_schur = Hxx_c - (Hlx_c.T @ t1 @ t0.T - t2 @ Hll_c @ t1 @ t0.T + t2 @ Hlx_c)
        grad[ind] = min_eig_vec.T @ grad_schur @ min_eig_vec
    return -1.0*min_eig_val, -1.0*grad

def scipy_minimize(inf_mats,H0, selection_init, k,num_poses):
    bounds = tuple([(0,1) for i in range(selection_init.shape[0])])
    cons = (
        { 'type': 'eq', 'fun': lambda x : np.sum(x) - k}
    )
    res = minimize(min_eig_obj_with_jac, selection_init, method='trust-constr', jac=True, args=(inf_mats,H0, num_poses),
                   constraints= cons, bounds=bounds, options={'disp': True})
    print(res.x)
    rounded_sol = roundsolution(res.x, k)
    print(rounded_sol)
    min_eig_val_unr, _, _ = find_min_eig_pair(inf_mats, res.x, H0, num_poses)
    min_eig_val_rounded, _, _ = find_min_eig_pair(inf_mats, rounded_sol, H0, num_poses)
    return rounded_sol,res.x,  min_eig_val_rounded, min_eig_val_unr

''' #########################################################'''

def find_min_eig_pair(inf_mats,selection, H0, num_poses):
    inds = np.where(selection > 1e-10)[0]
    #print(selection[inds])
    #final_inf_mat = np.sum(inf_mats[inds], axis=0)
    final_inf_mat = np.zeros(inf_mats[0].shape)
    for i in range(len(inds)):
        final_inf_mat = final_inf_mat + selection[inds[i]] * inf_mats[inds[i]]
    # add prior infomat H0
    final_inf_mat = final_inf_mat + H0
    H_schur = compute_schur_fim(final_inf_mat,num_poses )
    assert(utilities.check_symmetric(H_schur))
    #s_t = time.time()
    eigvals, eigvecs = la.eigh(H_schur)
    # e_t = time.time()
    # print("time taken to compute eigen vals and vectors dense: {:.6f}".format(e_t - s_t))
    #print(selection)
    # print("eigen vals")
    # print(eigvals[0:8])
    # print("null space")
    # print(scipy.linalg.null_space(H_schur).shape)
    return eigvals[0], eigvecs[:,0], final_inf_mat

def roundsolution(selection,k):
    idx = np.argpartition(selection, -k)[-k:]
    rounded_sol = np.zeros(len(selection))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol

def roundsolution_breakties(selection,k, all_mats, H0):
    s_rnd = np.round(selection, decimals=5)
    # print(np.argsort(s_rnd))
    # print(s_rnd[np.argsort(s_rnd)])
    all_eigs= []
    for m in all_mats:
        m_p = H0 + m
        assert (utilities.check_symmetric(m_p))
        eigvals, _ = la.eigh(m_p)
        all_eigs.append(eigvals[0])
    all_eigs = np.array(all_eigs)
    # print(all_eigs[np.argsort(s_rnd)])
    # print("----------------------------")

    zipped_vals = np.array([(s_rnd[i], all_eigs[i]) for i in range(len(s_rnd))], dtype=[('w', 'float'), ('weight', 'float')])
    idx = np.argpartition(zipped_vals, -k, order=['w', 'weight'])[-k:]
    rounded_sol = np.zeros(len(s_rnd))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol

def compute_info_metric(poses, points, meas, intrinsics, cands, selection, h_prior):
    num_poses = len(poses)
    num_points = len(points)
    h_full, graph, gtvals, poses_mask, points_mask = build_hfull(meas, points, poses, intrinsics, cands, selection)
    h_full = h_full + h_prior
    fim = compute_schur_fim(h_full, len(poses))
    least_fim_eig = np.linalg.eigvalsh(fim)[0]
    return least_fim_eig

def compute_rmse(measurements, poses, points, intrinsics, extr_cand,selected_inds,poses_with_noise, points_with_noise,h_prior_val, loc= False ):
    '''build the factor graph with the configuration '''
    graph_g, gtvals_g, poses_mask_g, points_mask_g = build_graph(measurements, poses, points, intrinsics,  extr_cand,
                                                                 selected_inds, rm_ill_posed = not loc)
    # Add a prior on pose x1. This indirectly specifies where the origin is.
    #same as the prior on the first pose given
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([h_prior_val,h_prior_val,h_prior_val,h_prior_val,h_prior_val,h_prior_val]))
    for i in range(len(poses)):
        if poses_mask_g[i] == 1:
            factor = PriorFactorPose3(X(i), poses[i], pose_noise)
            graph_g.push_back(factor)
            break
    for i in range(len(points)):
        if not loc:
            break
        if points_mask_g[i] == 1:
            point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
            factor = PriorFactorPoint3(L(i), points[i], point_noise)
            graph_g.push_back(factor)
            # if not loc:
            #     break
    # print("prior factor x0")
    ''' Give some initial values which are noisy'''
    initial_estimate_g = Values()
    n_poses_graph = 0
    n_points_graph = 0
    for val in gtvals_g.keys():
        '''
        If the resl variable is a pose, store the values
        '''
        if gtsam.Symbol(val).chr() == ord('x'):
            pose_ind = gtsam.Symbol(val).index()
            transformed_pose = poses_with_noise[pose_ind]
            # pose = gtvals.atPose3(val)
            # transformed_pose = pose.retract(0.1 * np.random.randn(6, 1))
            initial_estimate_g.insert(val, transformed_pose)
            n_poses_graph = n_poses_graph + 1
        elif gtsam.Symbol(val).chr() == ord('l'):
            point_ind = gtsam.Symbol(val).index()
            transformed_point = points_with_noise[point_ind]
            # point = gtvals.atPoint3(val)
            # transformed_point = point + 0.1 * np.random.randn(3)
            initial_estimate_g.insert(val, transformed_point)
            n_points_graph = n_points_graph + 1
    # print(poses_mask_g)
    print("num poses : " + str(n_poses_graph))
    print("num points : " + str(n_points_graph))
    # Optimize the graph and print results
    # params1 = gtsam.DoglegParams()
    # optimizer1 = DoglegOptimizer(graph_g, initial_estimate_g, params1)
    # try:
    #     result1 = optimizer1.optimize()
    # except Exception:
    #     result1 = Values()
    #     pass
    # # result.print('Final results:\n')
    # rmse1 = utilities.compute_traj_error(result1, poses, initial_estimate_g)
    # print("The RMSE of the estimated trajectory with best camera placement dogleg: " + str(rmse1))
    params = gtsam.LevenbergMarquardtParams()
    # params.setVerbosity('ERROR')
    optimizer = LevenbergMarquardtOptimizer(graph_g, initial_estimate_g, params)
    try:
        result = optimizer.optimize()
    except Exception:
        result = Values()
        pass
    # result.print('Final results:\n')
    rmse = utilities.compute_traj_error(result, poses, initial_estimate_g)
    print("The RMSE of the estimated trajectory with best camera placement: " + str(rmse))
    return rmse

# '''
# Generate a single dataset via simulation, run the algorithm for different
# select_k. log the results
#
# '''
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
#     parser.add_argument('-n', '--num_runs', help='number of runs in the experiment', default=25)
#     parser.add_argument('-s', '--select_k', help='number of cameras to select', default=2)
#     parser.add_argument('-t', '--traj_type', help='Type of trajectory 1:circle, 2:side, 3:forward, 4:random', default=1)
#     parser.add_argument('-o', '--output_dir', help='Output dir for output bag file', default='.')
#     parser.add_argument('-c', '--config_file',
#                         help='Yaml file which specifies the topic names, frequency of selection and time range',
#                         default='config/config.yaml')
#
#
#     args = parser.parse_args()
#
#     num_points = 30
#     num_poses =100
#     max_select_k = 7
#     K = Cal3_S2(100.0, 100.0, 0.0, 50.0, 50.0)
#     ''' Number of cameras to be selected'''
#     #select_k = args.select_k
#
#     traj = ""
#     ''' For a particular type of trajectory'''
#     for traj_ind in range(4, 5):
#         if traj_ind == 1:
#             traj = "circle"
#         elif traj_ind == 2:
#             traj = "side"
#         elif traj_ind == 3:
#             traj = "forward"
#         elif traj_ind >= 4:
#             traj = "random"
#
#         output_file_Path = os.path.join(args.output_dir, traj+"_{}runs".format(args.num_runs))
#         if os.path.exists(output_file_Path):
#             os.rmdir(output_file_Path)
#         os.mkdir(output_file_Path)
#
#         # Setup the parameters
#         # best_configs = {i: [] for i in range(2, max_select_k)}
#         # best_configs_fw = {i: [] for i in range(2, max_select_k)}
#
#         rmse_greedy_slam = {i: [] for i in range(2, max_select_k)}
#         rmse_greedy_loc = {i: [] for i in range(2, max_select_k)}
#         rmse_greedy_gt_slam = {i: [] for i in range(2, max_select_k)}
#         rmse_greedy_gt_loc = {i: [] for i in range(2, max_select_k)}
#
#         rmse_fw_slam = {i: [] for i in range(2, max_select_k)}
#         rmse_fw_loc = {i: [] for i in range(2, max_select_k)}
#         rmse_fw_gt_slam = {i: [] for i in range(2, max_select_k)}
#         rmse_fw_gt_loc = {i: [] for i in range(2, max_select_k)}
#
#         rmse_scipy_slam = {i: [] for i in range(2, max_select_k)}
#         rmse_scipy_loc = {i: [] for i in range(2, max_select_k)}
#         rmse_scipy_gt_slam = {i: [] for i in range(2, max_select_k)}
#         rmse_scipy_gt_loc = {i: [] for i in range(2, max_select_k)}
#
#         times_g = {i: [] for i in range(2, max_select_k)}
#         times_fw = {i: [] for i in range(2, max_select_k)}
#         times_scipy = {i: [] for i in range(2, max_select_k)}
#
#         iters_fw = {i: [] for i in range(2, max_select_k)}
#
#         greedy_scores = {i: [] for i in range(2, max_select_k)}
#         fw_scores = {i: [] for i in range(2, max_select_k)}
#         fw_scores_unrounded = {i: [] for i in range(2, max_select_k)}
#
#         scipy_scores = {i: [] for i in range(2, max_select_k)}
#         scipy_scores_unrounded = {i: [] for i in range(2, max_select_k)}
#
#         greedy_selected_cands = {i: [] for i in range(2, max_select_k)}
#         fw_selected_cands = {i: [] for i in range(2, max_select_k)}
#         scipy_selected_cands = {i: [] for i in range(2, max_select_k)}
#
#         fw_solution_unr_list = {i: [] for i in range(2, max_select_k)}
#         scipy_solution_unr_list = {i: [] for i in range(2, max_select_k)}
#
#
#         '''
#         EQUAL, RAND and STANDARD
#         '''
#         equal_scores = {i: [] for i in range(2, max_select_k)}
#         stnd_scores = {i: [] for i in range(2, max_select_k)}
#         rand_scores = {i: [] for i in range(2, max_select_k)}
#         equal_selected_cands = {i: [] for i in range(2, max_select_k)}
#         stnd_selected_cands = {i: [] for i in range(2, max_select_k)}
#         rand_selected_cands = {i: [] for i in range(2, max_select_k)}
#
#         rmse_equal_slam ={i: [] for i in range(2, max_select_k)}
#         rmse_equal_loc  ={i: [] for i in range(2, max_select_k)}
#         rmse_stnd_slam = {i: [] for i in range(2, max_select_k)}
#         rmse_stnd_loc  = {i: [] for i in range(2, max_select_k)}
#         rmse_rand_slam = {i: [] for i in range(2, max_select_k)}
#         rmse_rand_loc  = {i: [] for i in range(2, max_select_k)}
#
#
#         ''' Run a set of experiments'''
#         for exp_ind in range(args.num_runs):
#             ''' Generate the simulation data for greedy, franke-wolfe and RAND'''
#             poses, points, measurements, extr_cand, poses_with_noise, points_with_noise = generate_simulation_data(K, traj_ind, num_points, num_poses, False ) #args.traj_type
#
#             ''' write the data to log'''
#             utilities.write_data_logs(os.path.join(output_file_Path, "data_log_" + traj + "_{}.txt".format(exp_ind)),
#                                       poses, points, extr_cand, measurements, poses_with_noise,
#                                       points_with_noise)
#
#             ''' For this same data i.e poses and points, generate extr_cands and measurements for EQUAL and STANDARD'''
#             ''' the extr_cands for equal and stnd are a list corresponding to the selectKs as shown = [2,2,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6]'''
#             measurements_equal, extr_cand_equal = generate_meas_extr_EQUAL(poses, points)
#             measurements_stnd, extr_cand_stnd = generate_meas_extr_STANDARD(poses, points)
#             ''' write the data to log for EQUAL'''
#             utilities.write_data_logs(os.path.join(output_file_Path, "data_log_equal_" + traj + "_{}.txt".format(exp_ind)),
#                                       poses, points, extr_cand_equal, measurements_equal, poses_with_noise,
#                                       points_with_noise)
#             ''' write the data to log for STANDARD'''
#             utilities.write_data_logs(os.path.join(output_file_Path, "data_log_standard_" + traj + "_{}.txt".format(exp_ind)),
#                                       poses, points, extr_cand_stnd, measurements_stnd, poses_with_noise,
#                                       points_with_noise)
#
#             ''' Run optimization on different number of selected candidates'''
#             for select_k in range(2, max_select_k):
#                 '''
#                 Cases of Equal and Random and standard configurations
#                 '''
#                 ###EQUAL case candidates to evaluate
#                 b_i =0
#                 for f in range((select_k-1), 1, -1):
#                     b_i = b_i + f
#
#                 #selection_equal = extr_cand_equal[b_i: b_i + select_k]
#                 #selection_stnd = extr_cand_stnd[b_i: b_i + select_k]
#                 selection_equal = list(range(b_i , b_i + select_k))
#                 selection_stnd = list(range(b_i , b_i + select_k))
#                 selection_rand = random.sample(range(len(extr_cand)), select_k)
#
#
#                 score_equal = compute_info_metric(poses, points, measurements_equal, extr_cand_equal, selection_equal)
#                 score_stnd = compute_info_metric(poses, points, measurements_stnd, extr_cand_stnd, selection_stnd)
#                 score_rand = compute_info_metric(poses, points, measurements, extr_cand, selection_rand)
#
#                 equal_scores[select_k].append(score_equal)
#                 stnd_scores[select_k].append(score_stnd)
#                 rand_scores[select_k].append(score_rand)
#                 equal_selected_cands[select_k].append(selection_equal)
#                 stnd_selected_cands[select_k].append(selection_stnd)
#                 rand_selected_cands[select_k].append(selection_rand)
#
#                 print("RMSE for EQUAL, STANDARD and RAND")
#                 rmse_equal = compute_rmse(measurements_equal, poses, points, extr_cand_equal, selection_equal, poses_with_noise,
#                                           points_with_noise)
#                 rmse_equal_l = compute_rmse(measurements_equal, poses, points, extr_cand_equal, selection_equal,
#                                             poses_with_noise, points_with_noise, loc=True)
#
#                 rmse_equal_slam[select_k].append(rmse_equal)
#                 rmse_equal_loc[select_k].append(rmse_equal_l)
#
#                 rmse_stnd = compute_rmse(measurements_stnd, poses, points, extr_cand_stnd, selection_stnd,
#                                           poses_with_noise, points_with_noise)
#                 rmse_stnd_l = compute_rmse(measurements_stnd, poses, points, extr_cand_stnd, selection_stnd,
#                                             poses_with_noise, points_with_noise, loc=True)
#
#                 rmse_stnd_slam[select_k].append(rmse_stnd)
#                 rmse_stnd_loc[select_k].append(rmse_stnd_l)
#
#                 rmse_rand = compute_rmse(measurements, poses, points, extr_cand, selection_rand,
#                                           poses_with_noise,points_with_noise)
#                 rmse_rand_l = compute_rmse(measurements, poses, points, extr_cand, selection_rand,
#                                             poses_with_noise, points_with_noise, loc=True)
#
#                 rmse_rand_slam[select_k].append(rmse_rand)
#                 rmse_rand_loc[select_k].append(rmse_rand_l)
#
#
#                 #score, rmse, selected candidates
#
#                 ''' ###########################################################################################################'''
#                 best_score_g, best_config_g, selected_inds_g, time_greedy, best_score_fw,best_score_fw_unrounded,\
#                 best_config_fw, selected_inds_fw, solution_fw_unr, time_fw, num_iters_fw, best_score_scipy, best_score_scipy_unrounded,\
#                 selected_inds_scipy, solution_scipy_unr, time_scipy = run_single_experiment(poses, points, measurements, extr_cand, select_k)
#
#                 greedy_scores[select_k].append(best_score_g)
#                 fw_scores[select_k].append(best_score_fw)
#                 fw_scores_unrounded[select_k].append(best_score_fw_unrounded)
#                 scipy_scores[select_k].append(best_score_scipy)
#                 scipy_scores_unrounded[select_k].append(best_score_scipy_unrounded)
#
#                 greedy_selected_cands[select_k].append(selected_inds_g)
#                 fw_selected_cands[select_k].append(selected_inds_fw)
#                 scipy_selected_cands[select_k].append(selected_inds_scipy)
#
#                 fw_solution_unr_list[select_k].append(solution_fw_unr.tolist())
#                 scipy_solution_unr_list[select_k].append(solution_scipy_unr.tolist())
#
#                 times_g[select_k].append(time_greedy)
#                 times_fw[select_k].append(time_fw)
#                 times_scipy[select_k].append(time_scipy)
#                 iters_fw[select_k].append(num_iters_fw)
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
#                 rmse_greedy_slam[select_k].append(rmse_g)
#                 rmse_greedy_loc[select_k].append(rmse_g_loc)
#                 rmse_greedy_gt_slam[select_k].append(rmse_g_gt)
#                 rmse_greedy_gt_loc[select_k].append(rmse_g_gt_loc)
#
#                 print("RMSE for Franke-wolfe------------------------------------------------")
#                 rmse_fw = compute_rmse(measurements, poses, points, extr_cand, selected_inds_fw, poses_with_noise, points_with_noise)
#                 rmse_fw_l = compute_rmse(measurements, poses, points, extr_cand, selected_inds_fw, poses_with_noise, points_with_noise, loc=True)
#                 rmse_fw_gt = compute_rmse(measurements, poses, points, extr_cand, selected_inds_fw, poses, points)
#                 rmse_fw_gt_l = compute_rmse(measurements, poses, points, extr_cand, selected_inds_fw, poses, points, loc=True)
#                 rmse_fw_slam[select_k].append(rmse_fw)
#                 rmse_fw_loc[select_k].append(rmse_fw_l)
#                 rmse_fw_gt_slam[select_k].append(rmse_fw_gt)
#                 rmse_fw_gt_loc[select_k].append(rmse_fw_gt_l)
#
#                 print("RMSE for scipy------------------------------------------------")
#                 rmse_scipy = compute_rmse(measurements, poses, points, extr_cand, selected_inds_scipy, poses_with_noise,
#                                        points_with_noise)
#                 rmse_scipy_l = compute_rmse(measurements, poses, points, extr_cand, selected_inds_scipy, poses_with_noise,
#                                          points_with_noise, loc=True)
#                 rmse_scipy_gt = compute_rmse(measurements, poses, points, extr_cand, selected_inds_scipy, poses, points)
#                 rmse_scipy_gt_l = compute_rmse(measurements, poses, points, extr_cand, selected_inds_scipy, poses, points,
#                                             loc=True)
#                 rmse_scipy_slam[select_k].append(rmse_scipy)
#                 rmse_scipy_loc[select_k].append(rmse_scipy_l)
#                 rmse_scipy_gt_slam[select_k].append(rmse_scipy_gt)
#                 rmse_scipy_gt_loc[select_k].append(rmse_scipy_gt_l)
#
#                 visualize.reset()
#         for select_k in range(2, max_select_k):
#             rmse_errors_g = np.vstack((times_g[select_k], greedy_scores[select_k], rmse_greedy_slam[select_k], rmse_greedy_loc[select_k],rmse_greedy_gt_slam[select_k], rmse_greedy_gt_loc[select_k])).T
#             result_log_arr = np.hstack((np.array(greedy_selected_cands[select_k]), rmse_errors_g))
#
#             rmse_errors_fw = np.vstack((iters_fw[select_k], times_fw[select_k], fw_scores[select_k], fw_scores_unrounded[select_k], rmse_fw_slam[select_k], rmse_fw_loc[select_k], rmse_fw_gt_slam[select_k], rmse_fw_gt_loc[select_k])).T
#             result_log_arr = np.hstack((result_log_arr,np.array(fw_selected_cands[select_k]), np.array(fw_solution_unr_list[select_k]), rmse_errors_fw))
#
#             rmse_errors_scipy = np.vstack((times_scipy[select_k], scipy_scores[select_k], scipy_scores_unrounded[select_k], rmse_scipy_slam[select_k], rmse_scipy_loc[select_k],
#                                         rmse_scipy_gt_slam[select_k], rmse_scipy_gt_loc[select_k])).T
#             result_log_arr = np.hstack((result_log_arr, np.array(scipy_selected_cands[select_k]), np.array(scipy_solution_unr_list[select_k]), rmse_errors_scipy))
#
#             '''
#             EQUAL, RAND and STANDARD
#             '''
#             rmse_errors_equal = np.vstack((equal_scores[select_k],rmse_equal_slam[select_k], rmse_equal_loc[select_k])).T
#             result_log_arr = np.hstack((result_log_arr, np.array(equal_selected_cands[select_k]), rmse_errors_equal))
#
#             rmse_errors_stnd = np.vstack((stnd_scores[select_k], rmse_stnd_slam[select_k], rmse_stnd_loc[select_k])).T
#             result_log_arr = np.hstack((result_log_arr, np.array(stnd_selected_cands[select_k]), rmse_errors_stnd))
#
#             rmse_errors_rand = np.vstack((rand_scores[select_k], rmse_rand_slam[select_k], rmse_rand_loc[select_k])).T
#             result_log_arr = np.hstack((result_log_arr, np.array(rand_selected_cands[select_k]), rmse_errors_rand))
#
#
#             ''' Header : greedy_selection, time_greedy, greedy_score, rmse_slam_g, rmse_loc_g rmse_slam_gt_g, rmse_loc_gt_g,
#                          fw_selection, time_fw, fw_score, fw_score_unrounded, rmse_slam, rmse_loc, rmse_slam_gt, rmse_loc_gt  '''
#             np.savetxt(os.path.join(output_file_Path, "results_"+traj+"{}_.txt".format(select_k)), result_log_arr)
#
#     ''' visualize the final selection with franke wolfe'''
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