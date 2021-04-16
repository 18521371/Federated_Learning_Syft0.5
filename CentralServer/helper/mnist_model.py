import syft as sy


class FashionMNIST(sy.Module):
    def __init__(self, torch_ref):
        super().__init__(torch_ref=torch_ref)
        self.layer1 = self.torch_ref.nn.Conv2d(1, 4, 3, 1)
        self.layer2 = self.torch_ref.nn.Conv2d(4, 8, 3, 1)
        self.fc1 = self.torch_ref.nn.Linear(in_features=5*5*8, out_features=200)
        self.drop = self.torch_ref.nn.Dropout2d(0.25)
        self.fc2 = self.torch_ref.nn.Linear(in_features=200, out_features=10)

    def forward(self, X):
        out = self.layer1(X)
        out = self.torch_ref.nn.functional.relu(out)
        out = self.torch_ref.nn.functional.max_pool2d(out, 2, 2)
        out = self.layer2(out)
        out = self.torch_ref.nn.functional.relu(out)
        out = self.torch_ref.nn.functional.max_pool2d(out, 2, 2)
        out = out.view(64, 5*5*8)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.torch_ref.nn.functional.relu(out)
        out = self.fc2(out)
        return out



