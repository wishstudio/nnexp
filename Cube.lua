local Cube, parent = torch.class('nnexp.Cube','nn.Module')

function Cube:__init(b)
   parent.__init(self)
   self._squareInput = torch.Tensor()
end

function Cube:updateOutput(input)
   self._squareInput = torch.cmul(input, input)
   self.output = torch.cmul(input, self._squareInput)
   return self.output
end

function Cube:updateGradInput(input, gradOutput)
   self.gradInput = torch.cmul(gradOutput, self._squareInput)
   return self.gradInput
end