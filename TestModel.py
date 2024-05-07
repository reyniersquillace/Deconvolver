import torch
import numpy as np
import GenerateData

m = torch.load('./models/model_15.pt')
m.eval()
test_pulses, test_locs = GenerateData.generate_dummy(10, 1024)
y_preds = []
with torch.no_grad():
    for i in range(10):
        X_sample = test_pulses[i]
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
        y_pred = m(X_sample)[0].item()
        y_pred_err = m(X_sample)[1].item()
        y_preds.append(y_pred)
        print(f"Predicted y: {y_pred} +/- {y_pred_err}")
        print(f"Actual y: {test_locs[i]}")
        print("\n")
