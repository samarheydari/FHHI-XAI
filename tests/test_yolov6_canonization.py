import torch
import os
from LCRP.utils.galip_canonizers import YoloV6Canonizer

N=5
X=torch.randn((N,3,640,640), dtype=torch.float)
model=torch.load("models/best_v6s6_ckpt.pt")["model"]
model=model.to(torch.float).to("cpu")
if not os.path.exists("architecture.txt"):
    with open("architecture.txt","w") as f:
        f.write(str(model))
y_vanilla=model(X)
handles=YoloV6Canonizer().apply(model)
y_canonized=model(X)
for h in handles:
    h.remove()
y_final=model(X)

for output_selector in ["[0]","[1][0]","[1][1]","[1][2]","[1][3]"]:
    cnt=1
    for s in y_final[0].shape:
        cnt=cnt*s

    print("output",output_selector)
    print("canonized - vanilla: ", torch.norm(eval("y_canonized"+output_selector)-eval("y_vanilla"+output_selector))/cnt)
    print("uncanonized - canonized: ", torch.norm(eval("y_final"+output_selector)-eval("y_canonized"+output_selector))/cnt)
    print("uncanonized - vanilla: ", torch.norm(eval("y_final"+output_selector)-eval("y_vanilla"+output_selector))/cnt)
    print("\n==================================\n\n\n")
