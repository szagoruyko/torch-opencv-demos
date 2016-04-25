----------------------------------------------------------
-- NoiseFill 
----------------------------------------------------------
-- Fills last `num_noise_channels` channels of an existing `input` tensor with noise. 
local NoiseFill, parent = torch.class('nn.NoiseFill', 'nn.Module')

function NoiseFill:__init(num_noise_channels)
  parent.__init(self)

  -- last `num_noise_channels` maps will be filled with noise
  self.num_noise_channels = num_noise_channels
  self.mult = 1.0
end

function NoiseFill:updateOutput(input)
  self.output = self.output or input:new()
  self.output:resizeAs(input)


  -- copy non-noise part
  if self.num_noise_channels ~= input:size(2) then
    local ch_to_copy = input:size(2) - self.num_noise_channels
    self.output:narrow(2,1,ch_to_copy):copy(input:narrow(2,1,ch_to_copy))
  end

  -- fill noise
  if self.num_noise_channels > 0 then
    local num_channels = input:size(2)
    local first_noise_channel = num_channels - self.num_noise_channels + 1

    self.output:narrow(2,first_noise_channel, self.num_noise_channels):uniform():mul(self.mult)
  end
  return self.output
end

function NoiseFill:updateGradInput(input, gradOutput)
   self.gradInput:set(gradOutput)
   return self.gradInput
end

