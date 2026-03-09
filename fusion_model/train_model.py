import torch
import pandas as pd
from cognitive_model import CognitiveNet

data = pd.read_csv("../dataset/driver_features.csv")

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

X = torch.tensor(X,dtype=torch.float32)
y = torch.tensor(y)

model = CognitiveNet()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(100):

    outputs = model(X.unsqueeze(1))

    loss = criterion(outputs,y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("Epoch:",epoch,"Loss:",loss.item())

torch.save(model.state_dict(),"driver_model.pth")
