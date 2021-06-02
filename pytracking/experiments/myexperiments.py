from pytracking.evaluation import Tracker, get_dataset, trackerlist


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def uav_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('atom', 'default', range(1)) + \
               trackerlist('eco', 'default', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset

def transt():
    # Evaluate GOT on a circuit KYS I trained
    trackers = trackerlist('transt', 'default', range(1))
    dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

def transt_readout_test_encoder_mult():
    trackers = trackerlist('transt_readout_test_encoder_mult', 'default', range(1))
    dataset = get_dataset('got10k_val')  # , 'got10k_test')  # , 'got10k_val')  # , 'lasot')  # _val, 'got10k_ltrval')
    # dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

