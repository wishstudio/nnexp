require 'torch'
require 'xlua'
require 'nn'

-- create global nnexp table:
nnexp = {}

-- misc
torch.include('nnexp', 'Cube.lua')
