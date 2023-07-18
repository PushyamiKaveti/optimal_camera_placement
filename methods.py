from enum import Enum
import utilities
import visualize
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import List, Optional
import time
import gtsam
from gtsam.utils import plot
from gtsam import (DoglegOptimizer,
                         GenericProjectionFactorCal3_S2,
                         NonlinearFactorGraph, PinholeCameraCal3_S2, Point3,
                         PriorFactorPoint3, PriorFactorPose3,  Values)

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

class Metric(Enum):
    logdet = 1
    min_eig = 2
    mse = 3

def check_and_filter_points(poses, points, extrinsics, K, toplot=False):
    if toplot:
        plt.ion()
    fig1, ax1 = visualize.initialize_3d_plot(number=1, limits=np.array([[-30, 30], [-30, 30], [-30, 30]]),
                                                    view=[-30, -90])
    dict = {}
    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v
    # Simulated measurements from each camera pose, adding them to the factor graph
    for i, pose in enumerate(poses):
        for point in points:
            if toplot:
                plot.plot_point3_on_axes(ax1, point, 'ro')
        for k, comp_pose in enumerate(extrinsics):

            # compose pose with comp_pose T_wb * T_bc = T_wc, we get the pose of component camera W.R.T world
            pose_wc = pose.compose(comp_pose)
            if toplot:
                plot.plot_pose3_on_axes(ax1, pose_wc, axis_length=1, P=None, scale=1)
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
                        factor = GenericProjectionFactorCal3_S2(
                            measurement, measurement_noise, X(i), L(j), K, comp_pose)
                        if toplot:
                            plot.plot_point3_on_axes(ax1, point, 'b*')
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
        if toplot:
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            ax1.cla()
            #time.sleep(0.2)
            ax1.set_xlabel("X_axis")
            ax1.set_ylabel("Y_axis")
            ax1.set_zlabel("Z_axis")

    #plt.show()
    #This is not needed for localizaruion
    rm_indices = []
    # for k, v in dict.items():
    #     if v < 3:
    #         print("lm index: " + str(k) + ", lm value : " + str(points[k]))
    #         rm_indices.append(k)
    final_points_gtsam = []
    points_mask = np.ones(len(points))
    for i, pt in enumerate(points):
        if i not in rm_indices and i in dict.keys():
            final_points_gtsam.append(pt)
        else:
            points_mask[i] = 0
    #print(len(final_points_gtsam))
    return final_points_gtsam, points_mask


def error_pose(measurement: np.ndarray, K: np.ndarray, body_p_sensor:np.ndarray, point: np.ndarray,  this: gtsam.CustomFactor, values: gtsam.Values, J: Optional[List[np.ndarray]]):
    """
    This is he error function for custom factor for localization only given fixed landmarks
    """
    key = this.keys()[0]
    pos = values.atPose3(key)
    h0 = np.zeros((6, 6), order='F')
    h00 = np.zeros((6, 6), order='F')
    pose_wc = pos.compose(body_p_sensor, h0, h00)
    h0_hat = body_p_sensor.inverse().AdjointMap()
    Dpose = np.zeros((2, 6), order='F')
    Dpoint = np.zeros((2, 3), order='F')
    Dcal = np.zeros((2, 5), order='F')
    camera = PinholeCameraCal3_S2(pose_wc, K)
    measurement_hat = camera.project(point, Dpose, Dpoint, Dcal)
    error = measurement_hat - measurement
    if J is not None:
        J[0] = Dpose @ h0
    return error

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

    Hxx_schur = Hxx - Hlx.T @ np.linalg.inv(Hll) @ Hlx
    return Hxx_schur

def getMLE_multicam_loc(poses, points, extrinsics, K):

    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))

    dict = {}
    # Simulated measurements from each camera pose, adding them to the factor graph
    for i, pose in enumerate(poses):
        for k, comp_pose in enumerate(extrinsics):
            #compose pose with comp_pose T_wb * T_bc = T_wc, we get the pose of component camera W.R.T world
            pose_wc = pose.compose(comp_pose)
            camera = PinholeCameraCal3_S2(pose_wc, K)
           # print(pose)
            #print(comp_pose)
            #print(pose_wc)
            for j, point in enumerate(points):
                try:
                    measurement = camera.project(point)
                    if(measurement[0] > 1 and measurement[0] < (2*K.px()-2) and measurement[1] > 1 and measurement[1] < (2*K.py()-2)):
                        measurement = measurement + 1.0 * np.random.randn(2)
                        factor = gtsam.CustomFactor(measurement_noise, gtsam.KeyVector([X(i)]), partial(error_pose, measurement, K,comp_pose, point ))
                        graph.add(factor)

                        #print("projection factor x"+str(i)+" - l"+str(j))
                        if dict.get(i) is None:
                            dict[i] = [graph.nrFactors()-1]
                        else:
                            dict[i] = dict[i] + [graph.nrFactors()-1]
                    else:
                        #print("Measurement is out of bounds for L" + str(j))
                        #print(measurement)
                        pass

                except Exception:
                    #print("exception at L" + str(j))
                    pass
                    #print("Exception at Point")
                    #print(point)

    # for k, v in dict.items():
    #     print("Number of factors for pose: " + str(k)+" factors : "+str(len(v)))

    # rm_indices = []
    # rm_lm_indices=[]
    #
    # for k, v in dict.items():
    #     #print("lm index: " + str(k)+" factors : "+str(v))
    #     if len(v) < 2:
    #         print("lm index: " + str(k) + ", lm value : " + str(points[k]))
    #         rm_indices= rm_indices + v
    #         rm_lm_indices.append(k)
    #
    # for i in rm_indices:
    #     graph.remove(i)


    # Create the data structure to hold the initial estimate to the solution
    # Intentionally initialize the variables off from the ground truth
    initial_estimate = Values()
    for i, pose in enumerate(poses):
        transformed_pose = pose.retract(0.1 * np.random.randn(6, 1))
        initial_estimate.insert(X(i), transformed_pose)

    factor = PriorFactorPose3(X(0), poses[0], pose_noise)
    graph.add(factor)
    print("Number of factors in graph" + str(graph.nrFactors()))


    # # Optimize the graph and print results
    # params = gtsam.LevenbergMarquardtParams()
    # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    #
    # #params = gtsam.DoglegParams()
    # # params.setVerbosity('DELTA')
    # #optimizer = DoglegOptimizer(graph, initial_estimate, params)
    #
    # # params = gtsam.GaussNewtonParams()
    # # optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, params)
    #
    # # Optimize the factor graph
    # result = optimizer.optimize()
    # #print('Optimizing:')
    # try:
    #     result = optimizer.optimize()
    # except Exception:
    #     result = Values()
    #     pass
    # #result.print("result")
    result = Values()
    return result, graph


def getMLE_multicam(poses, points, extrinsics, K, points_mask):

    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
    #factor = PriorFactorPose3(X(0), poses[0], pose_noise)
    #graph.push_back(factor)
    #print("prior factor x0")
    dict = {}
    dict_lm_poses={}
    # Simulated measurements from each camera pose, adding them to the factor graph
    for i, pose in enumerate(poses):
        for j, point in enumerate(points):
            if not points_mask[j]:
                continue
            for k, comp_pose in enumerate(extrinsics):
                # compose pose with comp_pose T_wb * T_bc = T_wc, we get the pose of component camera W.R.T world
                pose_wc = pose.compose(comp_pose)
                camera = PinholeCameraCal3_S2(pose_wc, K)
                try:
                    measurement = camera.project(point)
                    # Check if the point is in the frustum of the camera
                    if (measurement[0] > 1 and measurement[0] < (2 * K.px() - 2) and measurement[1] > 1 and measurement[
                        1] < (2 * K.py() - 2)):
                        measurement = measurement + 1.0 * np.random.randn(2)
                        factor = GenericProjectionFactorCal3_S2(
                            measurement, measurement_noise, X(i), L(j), K, comp_pose)
                        graph.push_back(factor)

                        # print("projection factor x"+str(i)+" - l"+str(j))
                        #if tgl:
                        if dict.get(j) is None:
                            dict[j] = [graph.nrFactors() - 1]
                        else:
                            dict[j] = dict[j] + [graph.nrFactors() - 1]

                        if dict_lm_poses.get(j) is None:
                            dict_lm_poses[j] = {i}
                        else:
                            dict_lm_poses[j].add(i)
                           # tgl = False
                    else:
                        # print("Measurement is out of bounds for L" + str(j))
                        # print(measurement)
                        pass

                except Exception:
                    # print("exception at L" + str(j))
                    pass
                    # print("Exception at Point")
                    # print(point)

    rm_indices = []
    rm_lm_indices=[]
    for k, v in dict_lm_poses.items():
        #print("lm index: " + str(k)+" factors : "+str(v))
        if len(v) < 2:
            #print("lm index: " + str(k) + ", lm value : " + str(points[k]))
            rm_indices = rm_indices + dict[k]
            rm_lm_indices.append(k)
    #print(graph.keys())
    for i in rm_indices:
        graph.remove(i)
    #print(graph.keys())
    #print("Number of factors in graph"+str(graph.nrFactors()))
    ###graph.print('Factor Graph:\n')

    gt_vals = Values()
    # when we dont have enough constraints for poses.
    num_lms = 0
    num_ps = 0
    pose_mask= np.zeros(len(poses))
    for k in graph.keyVector():
        sym = gtsam.Symbol(k)
        # print(sym.chr())
        if sym.chr() == ord('x'):
            num_ps = num_ps + 1
            pose_idx = sym.index()
            gt_vals.insert(X(pose_idx), poses[pose_idx])
            #print('x'+str(pose_idx))
            pose_mask[pose_idx] = 1
        if sym.chr() == ord('l'):
            num_lms = num_lms + 1
            lm_idx = sym.index()
           # print("l" + str(lm_idx))

    # for i, pose in enumerate(poses):
    #     gt_vals.insert(X(i), pose)

    ins_prior = True
    point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    for j, point in enumerate(points):
        if j not in rm_lm_indices and j in dict.keys():
            if ins_prior:
                factor = PriorFactorPoint3(L(j), point, point_noise)
                # graph.push_back(factor)
                ins_prior = False
            gt_vals.insert(L(j), point)
            #print("l"+str(j))
        else:
            points_mask[j] = 0


    #print("num of poses in graph: "+ str(num_ps))
    #print("num of lms in graph: " + str(num_lms))
    if num_ps != len(poses):
        #print("not all poses have observed enough landmarks")
        pass
    #print(graph.keys())
    print("num of factors: "+str(graph.size()))
    print("Number of factors in graph" + str(graph.nrFactors()))
    return graph, gt_vals, pose_mask, points_mask

def greedy_selection(points, poses, K,pose_rots, pose_trans, Nc, metric= Metric.logdet, loc= False):
    num_rot_samples = len(pose_rots)
    num_trans_samples = len(pose_trans)
    avail_cand = np.ones((num_rot_samples * num_trans_samples, 1))
    last_best_extr = []

    best_extr = []
    best_score= 0.0
    best_graph = None
    best_gtvals= None
    #build the prior FIM
    num_poses = len(poses)
    num_points = len(points)
    h_prior = np.zeros((num_poses*6+num_points*3 , num_poses*6+num_points*3))
    h_prior[-num_poses*6: , -num_poses*6:] = np.eye(num_poses*6)
    h_prior[0 : -num_poses*6, 0: -num_poses*6: ] =  np.eye(num_points*3)
    # For each camera
    for i in range(0, Nc):
        max_inf = 0.0
        last_best_extr = best_extr
        selected_cand = 0
        for j, trans in enumerate(pose_trans):
            for k, rot in enumerate(pose_rots):
                if avail_cand[j*num_rot_samples + k, 0] == 1:
                    extr_cand = last_best_extr.copy()
                    c1 = gtsam.Pose3(gtsam.Rot3(rot), gtsam.Point3(trans[0], trans[1] ,trans[2]))
                    extr_cand.append(c1)
                    points_fil, points_mask = check_and_filter_points(poses, points, extr_cand, K, False)
                    if loc:
                        result, graph = getMLE_multicam_loc(poses, points_fil, extr_cand, K)
                    else:
                        graph, gtvals, poses_mask, points_mask = getMLE_multicam(poses, points, extr_cand, K,points_mask)
                    gt_vals = gtsam.Values()
                    if not loc:
                        gt_vals = gtvals
                    else:
                        for i1, pose in enumerate(poses):
                            gt_vals.insert(X(i1), pose)
                    fim, crlb = compute_CRLB(gt_vals, graph)
                    #build full info matrix
                    h_full = np.zeros((num_poses*6+num_points*3 , num_poses*6+num_points*3))
                    num_ps = np.count_nonzero(poses_mask)
                    num_lms = np.count_nonzero(points_mask)
                    tot_num_points = len(points)

                    '''
                    Extract all the block diagonal elements corresponding to the poses (bottom right square block)
                    into the full hessian
                    '''
                    pose_inds = []
                    fim_idx = 0
                    for val in gt_vals.keys():
                        sym = gtsam.Symbol(val)
                        if sym.chr() == ord('x'):
                            h_full_pose_idx = sym.index()
                            pose_inds.append(h_full_pose_idx)
                            h_full_pose_idx_start = tot_num_points*3 + h_full_pose_idx * 6
                            fim_pose_idx_start = num_lms * 3 + fim_idx * 6
                            h_full[ h_full_pose_idx_start : h_full_pose_idx_start + 6, h_full_pose_idx_start : h_full_pose_idx_start + 6 ] =  fim[fim_pose_idx_start: fim_pose_idx_start + 6 , fim_pose_idx_start: fim_pose_idx_start + 6]
                            fim_idx = fim_idx + 1

                    fim_idx=0
                    for val in gt_vals.keys():
                        sym = gtsam.Symbol(val)
                        if sym.chr() == ord('l'):
                            idx = sym.index()
                            h_full[idx*3: (idx+1)*3, idx*3: (idx+1)*3] = fim[fim_idx*3: (fim_idx+1)*3, fim_idx*3: (fim_idx+1)*3 ]
                            p_idx_fim =0
                            for pose_idx in pose_inds:
                                h_full_pose_idx_start = tot_num_points*3 + pose_idx*6
                                fim_pose_idx_start = num_lms * 3 + p_idx_fim * 6
                                h_full[idx*3: (idx+1)*3, h_full_pose_idx_start: h_full_pose_idx_start + 6] = fim[fim_idx*3: (fim_idx+1)*3, fim_pose_idx_start: fim_pose_idx_start + 6]
                                h_full[h_full_pose_idx_start: h_full_pose_idx_start + 6 , idx * 3: (idx + 1) * 3] = fim[fim_pose_idx_start: fim_pose_idx_start + 6 , fim_idx * 3: (fim_idx + 1) * 3]
                                p_idx_fim = p_idx_fim +1
                            fim_idx = fim_idx + 1
                    # if fim.shape == h_full.shape:
                    #     print(np.array_equal(fim, h_full))
                    assert (utilities.check_symmetric(h_full))
                    h_full = h_full + h_prior
                    least_fim_eig = 0.0
                    if not loc:
                        fim = compute_schur_fim(h_full ,len(poses))
                        #least_fim_eig = math.log(np.linalg.det(np.eye(fim.shape[0]) + fim))
                        if metric == Metric.logdet:
                            sign, least_fim_eig = np.linalg.slogdet(fim)
                            least_fim_eig = sign * least_fim_eig
                            # print(np.linalg.det(fim))
                            # print(least_fim_eig)
                            # print("-------------------------")
                        if metric == Metric.min_eig:
                            assert(utilities.check_symmetric(fim))
                            least_fim_eig = np.linalg.eigvalsh(fim)[0]
                    #least_fim_eig = compute_logdet(fim)

                    if least_fim_eig > max_inf:
                        max_inf = least_fim_eig
                        best_extr = extr_cand
                        selected_cand = j*num_rot_samples + k
                        best_graph = graph
                        best_gtvals = gt_vals
        best_score = max_inf
        avail_cand[selected_cand] = 0
        print("Best Score till now: "+ str(best_score))
        print("Next best Camera is: ")
        print(best_extr[-1])
        print("------------------")

    print("Selected candidates are : ")
    print(np.argwhere(avail_cand.flatten()==0))

    '''
    Compute the RMSE for the best camera placement
    '''
    ''' Give some initial values which are noisy'''
    initial_estimate = Values()
    for val in best_gtvals.keys():
        '''
        If the resl variable is a pose, store the values
        '''
        if gtsam.Symbol(val).chr() == ord('x'):
            pose = best_gtvals.atPose3(val)
            transformed_pose = pose.retract(0.1 * np.random.randn(6, 1))
            initial_estimate.insert(val, transformed_pose)
        elif gtsam.Symbol(val).chr() == ord('l'):
            point = best_gtvals.atPoint3(val)
            transformed_point = point + 0.1 * np.random.randn(3)
            initial_estimate.insert(val, transformed_point)

    # Optimize the graph and print results
    params = gtsam.DoglegParams()
    #params.setVerbosity('ERROR')
    optimizer = DoglegOptimizer(best_graph, initial_estimate, params)
    try:
        result = optimizer.optimize()
    except Exception:
        result = Values()
        pass
    #result.print('Final results:\n')
    rmse = utilities.compute_traj_error(result, poses)
    print("The RMSE of the estimated trajectory with best camera placement: "+ str(rmse))

    return best_extr, max_inf, avail_cand

