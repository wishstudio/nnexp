require 'torch'
require 'xlua'
require 'nn'

-- create global nnexp table:
nnexp = {}

-- c lib
require 'libnnexp'

-- misc
torch.include('nnexp', 'Cube.lua')
