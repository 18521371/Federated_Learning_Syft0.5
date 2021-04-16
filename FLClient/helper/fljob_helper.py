# Syft and Torch import
import syft as sy
from syft.federated.fl_client import FLClient
from syft.federated.fl_job import FLJob
from syft.federated.model_centric_fl_client import ModelCentricFLClient
from syft.util import get_root_data_path
import torch as th
from torchvision import datasets, transforms
from torch.autograd import Variable
# Third-party import
import numpy as np
#from loguru import logger
#import sys
# Utils
from . import *
from .config_helper import CYCLES_LOG, STATUS, MODEL_NAME, MODEL_VERSION, GRID_ADDRESS, CLIENT_ID, TRAIN_DATASET_PATH
from .data_helper import TrainLoader, TrainLoaderTest
from . import logger

# Called when client is accepted into FL cyclei
def on_accepted(job: FLJob):
    print("Accepted into cycle")
    #logger.info("Accepted into cycle")
    # Get parameters for perform training
    params = job.client_config
    batch_size = params["batch_size"]
    lr = params["lr"]

    # Load local train dataset
    train_data = TrainLoader(train_csv_path=TRAIN_DATASET_PATH, batch_size=batch_size)
    #train_data = TrainLoaderTest(batch_size)

    # Get training plan
    logger.info("Loading train dataset")
    training_plan = job.plans["training_plan"]      # Training plan (or get Inference plan for predict as well)
    model_params = job.model              # Model's parameters for training

    # For tracing
    losses = []
    
    # Start training using local train dataset  [MUST UPGRADE]
    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(train_data):
            data = Variable(data.view(64, 1, 28, 28))
            #data = Variable(data)
            target = th.nn.functional.one_hot(target,10)
            model_params, loss = training_plan(xs=data, ys=target, params=model_params)
            losses.append(loss.item())
            if batch_idx % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * batch_size, len(train_data) * batch_size,
                    100. * batch_idx / len(train_data), loss.item()))
            
            # For test
            if batch_idx > len(train_data) - 5:
                break
    
    # Report the updated parameters of the model to the PyGrid for aggregation
    job.report(model_params)

# Called when the client is rejected from cycle
def on_rejected(job: FLJob, timeout):
    if timeout is None:
        logger.info("Rejected from cycle without timeout (this means FL training is done)")
    else:
        logger.error("Rejected from cycle with timeout: {timeout}")
    
    STATUS["ended"] = True

# Called when error occured
def on_error(job: FLJob, error: Exception):
    logger.error("Error: {error}")

    STATUS["ended"] = True

# Create new job
def new_job(self) -> FLJob:
    # Check worker id
    if self.worker_id is None:
        auth_response = self.grid_worker.authenticate(
            self.auth_token, MODEL_NAME, MODEL_VERSION
        )
        self.worker_id = auth_response["data"]["worker_id"]
    
    # Create Federated Learning Job
    job = FLJob(
        worker_id=self,
        grid_worker=self.grid_worker,
        model_name = MODEL_NAME,
        model_version = MODEL_VERSION
    )
    return job

def create_client_and_run_cycle():
    logger.info("Grid address: {}".format(GRID_ADDRESS))
    # Connect client to PyGrid and create FLJob
    client = FLClient(url=GRID_ADDRESS, auth_token=None, secure=False)
    client.worker_id = client.grid_worker.authenticate(client.auth_token, MODEL_NAME, MODEL_VERSION)["data"]["worker_id"]
    job = client.new_job(MODEL_NAME, MODEL_VERSION)
    
    logger.info("WORKER ID: {} | {}".format(client.worker_id, CLIENT_ID))

    # Set event handlers
    job.add_listener(job.EVENT_ACCEPTED, on_accepted)
    job.add_listener(job.EVENT_REJECTED, on_rejected)
    job.add_listener(job.EVENT_ERROR, on_error)

    # Start handle!
    job.start()
