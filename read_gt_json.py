import gzip, json
from matplotlib import pyplot as plt

ep_id = str(5854)

with gzip.open(
    # "./data/trajectories_dirs/debug/trajectories.json.gz", "rt"
    "../dataset/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz", "r"
) as f:
    trajectories = json.load(f)
    print(trajectories[ep_id])
    ground_truth_move = trajectories[ep_id]['locations']
    plt.figure()
    plt.scatter([x[0] for x in ground_truth_move], [x[2] for x in ground_truth_move])
    plt.savefig("gt_trajectory.png")
    print("action number is ", len(trajectories[ep_id]['actions']))


print("=====================================")

with gzip.open(
    "../dataset/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz", "r"
) as f:
    trajectories = json.load(f)
    print(trajectories['episodes'][int(ep_id)-10])


 