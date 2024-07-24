from voiceguard.model.data import ModelId
from voiceguard.model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
import bittensor as bt
import constants
import argparse
import asyncio

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hotkey",
    type=str,
    help="The hotkey of the model to check",
)

bt.subtensor.add_args(parser)
args = parser.parse_args()
config = bt.config(parser)

subtensor = bt.subtensor(config=config)
subnet_uid = constants.SUBNET_UID
metagraph = subtensor.metagraph(subnet_uid)

wallet = None
model_metadata_store = ChainModelMetadataStore(subtensor, subnet_uid, wallet)

model_name = asyncio.run(model_metadata_store.retrieve_model_metadata(args.hotkey))

print(model_name)