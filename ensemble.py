import os

from deepstack.base import Member
from deepstack.ensemble import StackEnsemble

# Load base-learners
model = Member(name="ResNet50")
model.load(os.getcwd() + '/models/resnet50.pth')
model.eval()

# Create ensemble and add members
stack = StackEnsemble()
stack.add_member(model)
stack.fit()
stack.describe()
stack.predict()