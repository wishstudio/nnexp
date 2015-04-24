local Cube, parent = torch.class('nn.Cube','nn.Module')

function Cube:__init(b)
   parent.__init(self)
end

function Cube:updateOutput(input)
   return input.nn.Cube_updateOutput(self, input)
end

function Cube:updateGradInput(input, gradOutput)
   return input.nn.Cube_updateGradInput(self, input, gradOutput)
end