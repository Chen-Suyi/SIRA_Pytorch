import numpy as np
import open3d
from sklearn.neighbors import NearestNeighbors
from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE


# ICP
#####################################################################################
def best_fit_transform(src_xyz, ref_xyz):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert src_xyz.shape == ref_xyz.shape

    # get number of dimensions
    m = src_xyz.shape[1]

    # translate points to their centroids
    src_xyz_centroid = np.mean(src_xyz, axis=0)
    ref_xyz_centroid = np.mean(ref_xyz, axis=0)
    src_xyz = src_xyz - src_xyz_centroid
    ref_xyz = ref_xyz - ref_xyz_centroid

    # rotation matrix
    H = np.dot(src_xyz.T, ref_xyz)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = ref_xyz_centroid.T - np.dot(R, src_xyz_centroid.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src_xyz, ref_xyz):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src_xyz.shape == ref_xyz.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(ref_xyz)
    distances, indices = neigh.kneighbors(src_xyz, return_distance=True)
    return distances.ravel(), indices.ravel()


def ICP(src_xyz, ref_xyz, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert src_xyz.shape == ref_xyz.shape

    # get number of dimensions
    m = src_xyz.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src_xyz_iter = np.ones((m + 1, src_xyz.shape[0]))
    ref_xyz_iter = np.ones((m + 1, ref_xyz.shape[0]))
    src_xyz_iter[:m, :] = np.copy(src_xyz.T)
    ref_xyz_iter[:m, :] = np.copy(ref_xyz.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src_xyz_iter = np.dot(init_pose, src_xyz_iter)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src_xyz_iter[:m, :].T,
                                              ref_xyz_iter[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src_xyz_iter[:m, :].T,
                                     ref_xyz_iter[:m, indices].T)

        # update the current source
        src_xyz_iter = np.dot(T, src_xyz_iter)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(src_xyz, src_xyz_iter[:m, :].T)

    return T


# Go-ICP
#####################################################################################
def Go_ICP(src_xyz, ref_xyz):

    def loadPointCloud(xyz_np):
        plist = xyz_np.tolist()
        p3dlist = []
        for x, y, z in plist:
            pt = POINT3D(x, y, z)
            p3dlist.append(pt)
        return xyz_np.shape[0], p3dlist

    src_xyz_centroid = np.mean(src_xyz, axis=0)
    ref_xyz_centroid = np.mean(ref_xyz, axis=0)
    src_xyz = src_xyz - src_xyz_centroid
    ref_xyz = ref_xyz - ref_xyz_centroid
    Nm, src_points = loadPointCloud(src_xyz)
    Nd, ref_points = loadPointCloud(ref_xyz)
    goicp = GoICP()
    rNode = ROTNODE()
    tNode = TRANSNODE()
    rNode.a = -3.1416
    rNode.b = -3.1416
    rNode.c = -3.1416
    rNode.w = 6.2832

    tNode.x = -0.5
    tNode.y = -0.5
    tNode.z = -0.5
    tNode.w = 1.0

    goicp.MSEThresh = 0.001
    goicp.trimFraction = 0.25

    if (goicp.trimFraction < 0.001):
        goicp.doTrim = False
    # goicp.loadModelAndData(Nm, src_points, Nd, ref_points)
    goicp.loadModelAndData(Nd, ref_points, Nm, src_points)
    goicp.setDTSizeAndFactor(300, 2.0)
    goicp.BuildDT()
    goicp.Register()
    R = goicp.optimalRotation(
    )  # A python list of 3x3 is returned with the optimal rotation
    t = goicp.optimalTranslation(
    )  # A python list of 1x3 is returned with the optimal translation
    T = np.identity(4)
    T[:3, :3] = np.array(R)
    T[:3, 3] = np.array(t) - np.dot(R, src_xyz_centroid.T) + ref_xyz_centroid.T
    return T


# FGR
#####################################################################################
def FGR(src_xyz, ref_xyz, voxel_size=0.033):
    src_pcd, ref_pcd = open3d.geometry.PointCloud(
    ), open3d.geometry.PointCloud()

    src_pcd.points = open3d.utility.Vector3dVector(src_xyz)
    ref_pcd.points = open3d.utility.Vector3dVector(ref_xyz)

    radius_normal = voxel_size * 2  # kdtree参数，用于估计法线的半径，一般设为体素大小的2倍
    src_pcd.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,
                                                max_nn=30))
    ref_pcd.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,
                                                max_nn=30))
    radius_feature = voxel_size * 5  # kdtree参数，用于估计特征的半径，设为体素大小的5倍

    src_fpfh = open3d.registration.compute_fpfh_feature(
        src_pcd,
        open3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100))  # 计算特征的2个参数，下采样的点云数据，搜索方法kdtree
    ref_fpfh = open3d.registration.compute_fpfh_feature(
        ref_pcd,
        open3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100))  # 计算特征的2个参数，下采样的点云数据，搜索方法kdtree

    # 执行配准
    # distance_threshold = voxel_size * 1.5  # 设定距离阈值为体素的1.5倍
    # 2个降采样的点云，两个点云的特征，距离阈值，一个函数，4，一个list[0.9的两个对应点的线段长度阈值，两个点的距离阈值]，一个函数设定最大迭代次数和最大验证次数
    # result = open3d.registration.registration_ransac_based_on_feature_matching(
    #     src_pcd, ref_pcd, src_fpfh, ref_fpfh, distance_threshold, open3d.registration.TransformationEstimationPointToPoint(False), 4, [
    #         open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #         open3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    #     ], open3d.registration.RANSACConvergenceCriteria(4000000, 500))
    distance_threshold = voxel_size * 1.5
    result = open3d.registration.registration_fast_based_on_feature_matching(
        src_pcd, ref_pcd, src_fpfh, ref_fpfh,
        open3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    transform_pred = result.transformation.copy()

    return transform_pred


# FPFH + RANSAC
#####################################################################################
def FPFH_RANSAC(src_xyz, ref_xyz, voxel_size=0.03):
    src_pcd, ref_pcd = open3d.geometry.PointCloud(
    ), open3d.geometry.PointCloud()

    src_pcd.points = open3d.utility.Vector3dVector(src_xyz)
    ref_pcd.points = open3d.utility.Vector3dVector(ref_xyz)

    radius_normal = voxel_size * 2  # kdtree参数，用于估计法线的半径，一般设为体素大小的2倍
    src_pcd.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,
                                                max_nn=30))
    ref_pcd.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,
                                                max_nn=30))
    radius_feature = voxel_size * 5  # kdtree参数，用于估计特征的半径，设为体素大小的5倍

    src_fpfh = open3d.registration.compute_fpfh_feature(
        src_pcd,
        open3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100))  # 计算特征的2个参数，下采样的点云数据，搜索方法kdtree
    ref_fpfh = open3d.registration.compute_fpfh_feature(
        ref_pcd,
        open3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100))  # 计算特征的2个参数，下采样的点云数据，搜索方法kdtree

    # 执行配准
    distance_threshold = voxel_size * 1.5  # 设定距离阈值为体素的1.5倍
    # 2个降采样的点云，两个点云的特征，距离阈值，一个函数，4，一个list[0.9的两个对应点的线段长度阈值，两个点的距离阈值]，一个函数设定最大迭代次数和最大验证次数
    result = open3d.registration.registration_ransac_based_on_feature_matching(
        src_pcd, ref_pcd, src_fpfh, ref_fpfh, distance_threshold,
        open3d.registration.TransformationEstimationPointToPoint(False), 4, [
            open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], open3d.registration.RANSACConvergenceCriteria(4000000, 500))
    transform_pred = result.transformation.copy()

    return transform_pred
