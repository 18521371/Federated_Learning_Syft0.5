# For connect to PyGrid
class ClientConfig():
    def __init__(self, grid_address, model_name, model_version, client_id, train_dataset_path):
        self.grid_address = grid_address
        self.model_name = model_name
        self.model_version = model_version
        self.client_id = client_id
        self.train_dataset_path = train_dataset_path
    
    def getModelName(self):
        return self.model_name
    def getModelVersion(self):
        return self.model_version
    def getGridAddr(self):
        return self.grid_address
    def getClientId(self):
        return self.client_id
    def getPath(self):
        return self.train_dataset_path

GRID_ADDRESS = ""
MODEL_NAME = ""
MODEL_VERSION = ""
CLIENT_ID = ""

# Train dataset path
TRAIN_DATASET_PATH = ""

# For tracing
CYCLES_LOG = []
STATUS = {
    "ended": False
}
