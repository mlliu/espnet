# change the code from applying kmeans to applying PCA


import argparse
import logging
import os
import random
import sys

import joblib
import numpy as np
#from sklearn.cluster import MiniBatchKMeans
import faiss

from espnet.utils.cli_readers import file_reader_helper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_path", type=str, required=True)
    parser.add_argument("--n_components", type=int, required=True, help="dim for pca reduction")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    )
    parser.add_argument(
        "--eigen_power", type=float, default=0, help="eigen power, -0.5 for whitening"
    )

    parser.add_argument(
        "--in_filetype",
        type=str,
        default="sound",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "rspecifier",
        type=str,
        nargs="+",
        help="Read specifier for feats. e.g. ark:some.ark",
    )
    return parser





def load_feature_shard(rspecifier, in_filetype, percent):
    feats = []
    for utt, feat in file_reader_helper(rspecifier, in_filetype):
        feats.append(feat)
    if percent < 0:
        return np.concatenate(feats, axis=0)
    else:
        nsample = int(np.ceil(len(feats) * percent))
        sampled_feat = random.sample(feats, nsample)
        sampled_feat = np.concatenate(
            sampled_feat,
            axis=0,
        )
        logger.info(
            (
                f"sampled {nsample} utterances, {len(sampled_feat)} frames "
                f"from rspecifier {rspecifier}"
            )
        )
        return sampled_feat


def load_feature(rspecifiers, in_filetype, percent):
    assert percent <= 1.0
    if not isinstance(rspecifiers, list):
        rspecifiers = [rspecifiers]
    feat = np.concatenate(
        [
            load_feature_shard(rspecifier, in_filetype, percent)
            for rspecifier in rspecifiers
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_pca(
    rspecifier,
    in_filetype,
    pca_path,
    n_components,
    seed,
    percent,
    eigen_power
):
    np.random.seed(seed)
    feat = load_feature(rspecifier, in_filetype, percent)
    # km_model = get_km_model(
    #     n_clusters,
    #     init,
    #     max_iter,
    #     batch_size,
    #     tol,
    #     max_no_improvement,
    #     n_init,
    #     reassignment_ratio,
    #     seed,
    # )
    # km_model.fit(feat)
    # joblib.dump(km_model, km_path)
    #
    # inertia = -km_model.score(feat) / len(feat)
    # logger.info("total intertia: %.5f", inertia)
    # logger.info("finished successfully")
    print("Computing PCA")
    pca = faiss.PCAMatrix(feat.shape[-1], n_components, eigen_power)
    pca.train(feat)
    b = faiss.vector_to_array(pca.b)
    A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)

    # save the pca matrix
    prefix = str(n_components)
    if eigen_power != 0:
        prefix += f"_{eigen_power}"
    np.save(f"{pca_path}/{prefix}_pca_A", A.T)
    np.save(f"{pca_path}/{prefix}_pca_b", b)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(str(args))

    learn_pca(**vars(args))
