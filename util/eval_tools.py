import numpy as np
import matplotlib.pyplot as plt


def gen_sim_mat(class_list1, class_list2):
    """
    Generates a similarity matrix
    Args:
        class_list1: row-ordered class indicators
        class_list2:
    Returns: [N1 N2]
    """
    c1 = np.asarray(class_list1)
    c2 = np.asarray(class_list2)
    sim_mat = np.matmul(c1, c2.T)
    sim_mat[sim_mat > 0] = 1
    return sim_mat


def compute_hamming_dist(a, b):
    """
    Computes hamming distance vector wisely
    Args:
        a: row-ordered codes {0,1}
        b: row-ordered codes {0,1}
    Returns:
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a_m1 = a - 1
    b_m1 = b - 1
    c1 = np.matmul(a, b_m1.T)
    c2 = np.matmul(a_m1, b.T)
    return np.abs(c1 + c2)


def eval_cls_map(query, target, cls1, cls2, at=None, computePR=False):
    """
    Mean average precision computation
    :param query:
    :param target:
    :param cls1:
    :param cls2:
    :param at:
    :return:
    """
    sim_mat = gen_sim_mat(cls1, cls2)
#    test_sim = np.sum(sim_mat, axis=1)
#    print(len(test_sim))
#    print(np.sum(test_sim))
    query_size = query.shape[0]
    distances = compute_hamming_dist(query, target)
    dist_argsort = np.argsort(distances)

    map_count = 0.
    average_precision = 0.
    _precision = 0.
    total_items = dist_argsort.shape[1]
    top_k = at if at is not None else total_items
    if not computePR: total_items = top_k
    pr_count = 0

    pr_curve = np.zeros((2, total_items))

    for i in range(query_size):
        gt_count = 0.
        precision = 0.
        corr_items = 0.
        # count_size = 0 if at is None else at
        # for j in range(dist_argsort.shape[1]):
        #     this_ind = dist_argsort[i, j]
        #     if sim_mat[i, this_ind] == 1:
        #         gt_count += 1.
        #         precision += gt_count / (j + 1.)
        #
        #     if gt_count >= count_size > 0:
        #         break
        num_gt = len(np.where(sim_mat[i,:]>0)[0])
        pr_count += (num_gt>0)
        if(num_gt==0):
            print("Error, num_gt is {}, i is {}".format(num_gt, i))
            print(cls1[i])
            print(sim_mat[i,:])
        for j in range(total_items):
            this_ind = dist_argsort[i, j]
            if sim_mat[i, this_ind] > 0:
                corr_items += 1
                if j<top_k:
                    gt_count += 1.
                    precision += gt_count / (j + 1.)
            if(num_gt>0):
                pr_curve[0,j] += corr_items / (j+1)
                pr_curve[1,j] += corr_items / num_gt
        if gt_count > 0:
            average_precision += precision / gt_count
            map_count += 1.
            _precision += gt_count / top_k
    pr_curve = pr_curve[:, pr_curve[0].argsort()]
    pr_curve /= pr_count

    return average_precision / map_count, _precision / map_count,pr_curve
