import subprocess


# for n_experts in [1, 2]:
#     subprocess.call(
#         f"python src/main.py --clf-type moe --reg-coef 0.3 --n-experts {n_experts} "
#         f"--learning-rate 0.005 --batch-size 64 --n-concepts 2 --env-id CarRacing-v2 "
#         f"--mode nmf --decay-factor 0.98 --max-patience 10 --save-start 0 --n-clf 1",
#         shell=True
#     )


# for n_experts in [1]:
#     subprocess.call(
#         f"python src/main.py --clf-type moe --reg-coef 0.008 --n-experts {n_experts} "
#         f"--learning-rate 0.005 --batch-size 64 --n-concepts 5 --env-id MsPacmanNoFrameskip-v4 "
#         f"--mode nmf --decay-factor 0.98 --max-patience 10 --save-start 0 --n-clf 1",
#         shell=True
#     )


# for n_experts in [1,]:
#     subprocess.call(
#         f"python src/main.py --clf-type moe --reg-coef 0.01 --n-experts {n_experts} "
#         f"--learning-rate 0.001 --batch-size 64 --n-concepts 3 --env-id BreakoutNoFrameskip-v4 "
#         f"--mode nmf --decay-factor 0.98 --max-patience 10 --save-start 0 --n-clf 1",
#         shell=True
#     )

# for n_experts in [3, 4, 5, 6, 7]:
#     subprocess.call(
#         f"python src/main.py --clf-type moe-kmeans --reg-coef 0.3 --n-experts {n_experts} "
#         f"--learning-rate 0.001 --batch-size 64 --n-concepts 2 --env-id PongNoFrameskip-v4 "
#         f"--mode nmf --decay-factor 0.98 --max-patience 10 --save-start 0 --n-clf 1",
#         shell=True
#     )


# for n_experts in [2, ]:
#     subprocess.call(
#         f"python src/main.py --clf-type moe --reg-coef 0.3 --n-experts {n_experts} "
#         f"--learning-rate 0.001 --batch-size 64 --n-concepts 2 --env-id PongNoFrameskip-v4 "
#         f"--mode nmf --decay-factor 0.98 --max-patience 10 --save-start 0 --n-clf 1",
#         shell=True
#     )


# for n_experts in [2, 3]:
#     subprocess.call(
#         f"python src/main.py --clf-type moe --reg-coef 0.008 --n-experts {n_experts} "
#         f"--learning-rate 0.005 --batch-size 64 --n-concepts 5 --env-id MsPacmanNoFrameskip-v4 "
#         f"--mode nmf --decay-factor 0.98 --max-patience 10 --save-start 0 --n-clf 1",
#         shell=True
#     )


# for n_experts in [2, 3]:
#     subprocess.call(
#         f"python src/main.py --clf-type moe --reg-coef 0.01 --n-experts {n_experts} "
#         f"--learning-rate 0.001 --batch-size 64 --n-concepts 3 --env-id BreakoutNoFrameskip-v4 "
#         f"--mode nmf --decay-factor 0.98 --max-patience 10 --save-start 0 --n-clf 1",
#         shell=True
#     )

# for args in [
#     ["black-box", 0, "PongNoFrameskip-v4"],
#     ["moe", 0, "PongNoFrameskip-v4"],
#     ["moe", 1, "PongNoFrameskip-v4"],
#     ["pwnet", 0, "PongNoFrameskip-v4"],
#     ["tree", 0, "PongNoFrameskip-v4"],

#     ["black-box", 0, "BreakoutNoFrameskip-v4"],
#     ["moe", 0, "BreakoutNoFrameskip-v4"],
#     ["moe", 1, "BreakoutNoFrameskip-v4"],
#     ["pwnet", 0, "BreakoutNoFrameskip-v4"],
#     ["tree", 0, "BreakoutNoFrameskip-v4"],

#     ["black-box", 0, "MsPacmanNoFrameskip-v4"],
#     ["moe", 0, "MsPacmanNoFrameskip-v4"],
#     ["moe", 1, "MsPacmanNoFrameskip-v4"],
#     ["pwnet", 0, "MsPacmanNoFrameskip-v4"],
#     ["tree", 0, "MsPacmanNoFrameskip-v4"]
# ]:
#     subprocess.call(
#         f"python src/eval.py --method {args[0]} --version {args[1]} --env-id {args[2]}", shell=True)


for env_id in ["CarRacing-v2",]:
    for i in range(37, 38):
        subprocess.call(
            f"python src/eval.py --method moe --version {i} --env-id {env_id}", shell=True)
