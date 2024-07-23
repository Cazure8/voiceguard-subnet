import torch
import random
import bittensor as bt
import functools
import multiprocessing
from typing import Any, Tuple

from voiceguard.model.data import ModelId
from typing import List


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_random_uids(
    self, k: int, exclude: List[int] = None
) -> torch.LongTensor:
    # """Returns k available random uids from the metagraph.
    # Args:
    #     k (int): Number of uids to return.
    #     exclude (List[int]): List of uids to exclude from the random sampling.
    # Returns:
    #     uids (torch.LongTensor): Randomly sampled available uids.
    # Notes:
    #     If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    # """
    # candidate_uids = []
    # avail_uids = []

    # for uid in range(self.metagraph.n.item()):
    #     if uid == self.uid:
    #         continue
    #     uid_is_available = check_uid_availability(
    #         self.metagraph, uid, self.config.neuron.vpermit_tao_limit
    #     )
    #     uid_is_not_excluded = exclude is None or uid not in exclude

    #     if uid_is_available:
    #         avail_uids.append(uid)
    #         if uid_is_not_excluded:
    #             candidate_uids.append(uid)

    # # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    # available_uids = candidate_uids
    # if len(candidate_uids) < k:
    #     available_uids += random.sample(
    #         [uid for uid in avail_uids if uid not in candidate_uids],
    #         k - len(candidate_uids),
    #     )
    # uids = torch.tensor(random.sample(available_uids, k))
    # return uids
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available and uid_is_not_excluded:
            candidate_uids.append(uid)
        elif uid_is_available:
            avail_uids.append(uid)

    # If not enough candidate_uids, supplement from avail_uids, ensuring they're not in exclude list
    if len(candidate_uids) < k:
        additional_uids_needed = k - len(candidate_uids)
        filtered_avail_uids = [uid for uid in avail_uids if uid not in exclude]
        additional_uids = random.sample(
            filtered_avail_uids, min(additional_uids_needed, len(filtered_avail_uids))
        )
        candidate_uids.extend(additional_uids)

    # Safeguard against trying to sample more than what is available
    num_to_sample = min(k, len(candidate_uids))
    
    uids = random.sample(candidate_uids, num_to_sample)
    bt.logging.debug(f"returning available uids: {uids}")
    return uids


def assert_registered(wallet: bt.wallet, metagraph: bt.metagraph) -> int:
    """Asserts the wallet is a registered miner and returns the miner's UID.

    Raises:
        ValueError: If the wallet is not registered.
    """
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"You are not registered. \nUse: \n`btcli s register --netuid {metagraph.netuid}` to register via burn \n or btcli s pow_register --netuid {metagraph.netuid} to register with a proof of work"
        )
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.success(
        f"You are registered with address: {wallet.hotkey.ss58_address} and uid: {uid}"
    )

    return uid


def validate_hf_repo_id(repo_id: str) -> Tuple[str, str]:
    """Verifies a Hugging Face repo id is valid and returns it split into namespace and name.

    Raises:
        ValueError: If the repo id is invalid.
    """

    if not repo_id:
        raise ValueError("Hugging Face repo id cannot be empty.")

    if not 3 < len(repo_id) <= ModelId.MAX_REPO_ID_LENGTH:
        raise ValueError(
            f"Hugging Face repo id must be between 3 and {ModelId.MAX_REPO_ID_LENGTH} characters."
        )

    parts = repo_id.split("/")
    if len(parts) != 2:
        raise ValueError(
            "Hugging Face repo id must be in the format <org or user name>/<repo_name>."
        )

    return parts[0], parts[1]


# def run_in_subprocess(func: functools.partial, ttl: int) -> Any:
#     """Runs the provided function on a subprocess with 'ttl' seconds to complete.

#     Args:
#         func (functools.partial): Function to be run.
#         ttl (int): How long to try for in seconds.

#     Returns:
#         Any: The value returned by 'func'
#     """

#     def wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
#         try:
#             result = func()
#             queue.put(result)
#         except (Exception, BaseException) as e:
#             # Catch exceptions here to add them to the queue.
#             queue.put(e)

#     # Use "fork" (the default on all POSIX except macOS), because pickling doesn't seem
#     # to work on "spawn".
#     ctx = multiprocessing.get_context("fork")
#     queue = ctx.Queue()
#     process = ctx.Process(target=wrapped_func, args=[func, queue])

#     process.start()

#     process.join(timeout=ttl)

#     if process.is_alive():
#         process.terminate()
#         process.join()
#         raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

#     # Raises an error if the queue is empty. This is fine. It means our subprocess timed out.
#     result = queue.get(block=False)

#     # If we put an exception on the queue then raise instead of returning.
#     if isinstance(result, Exception):
#         raise result
#     if isinstance(result, BaseException):
#         raise Exception(f"BaseException raised in subprocess: {str(result)}")

#     return result
