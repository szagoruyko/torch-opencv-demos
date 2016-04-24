local SpatialCircularPadding, parent = torch.class('nn.SpatialCircularPadding', 'nn.Module')

function SpatialCircularPadding:__init(pad_l, pad_r, pad_t, pad_b)
   parent.__init(self)
   self.pad_l = pad_l
   self.pad_r = pad_r or self.pad_l
   self.pad_t = pad_t or self.pad_l
   self.pad_b = pad_b or self.pad_l

   if self.pad_l < 0 or  self.pad_r < 0 or self.pad_t < 0 or self.pad_b < 0 then
      error('padding should be > 0')
   end
end

function SpatialCircularPadding:updateOutput(input)
   if input:dim() ~= 4 and  input:dim() ~= 3 then
      error('input must be 3 or 4-dimensional')
   end

   local ac = 4 - input:dim()

   -- sizes
   local h = input:size(3 - ac) + self.pad_t + self.pad_b
   local w = input:size(4- ac) + self.pad_l + self.pad_r
   
   if w < 1 or h < 1 then error('input is too small') end
   
   if input:dim() == 4 then
      self.output:resize(input:size(1), input:size(2), h, w)
   else
      self.output:resize(input:size(1), h, w)
   end
   -- self.output:zero()
   
   -- crop input if necessary
   local c_input = input
  
   -- crop outout if necessary
   local c_output = self.output
   c_output = c_output:narrow(3- ac, 1 + self.pad_t, c_output:size(3- ac) - self.pad_t)
   c_output = c_output:narrow(3- ac, 1, c_output:size(3- ac) - self.pad_b)
   c_output = c_output:narrow(4- ac, 1 + self.pad_l, c_output:size(4- ac) - self.pad_l)
   c_output = c_output:narrow(4- ac, 1, c_output:size(4- ac) - self.pad_r)
   
   -- copy input to output
   c_output:copy(c_input)

   -----------------------------------------------------------------------
   -- It should be done like folowing, but it is not clear about corners, 
   -- Filling them with 0 is bad idea, since NN will find the corners then 
   -- So use a little weird version
   ------------------------------------------------------------------------

   -- local tb_slice = self.output:narrow(4, self.pad_l+1,  input:size(4))
   -- local lr_slice = self.output:narrow(3, self.pad_t+1,  input:size(3))

   -- tb_slice:narrow(3, 1, self.pad_t):copy(input:narrow(3, input:size(3) - self.pad_t + 1, self.pad_t))
   -- tb_slice:narrow(3, input:size(3) + self.pad_t + 1, self.pad_b):copy(input:narrow(3, 1, self.pad_b))

   -- lr_slice:narrow(4, 1, self.pad_l):copy(input:narrow(4, input:size(4) - self.pad_l + 1, self.pad_l))
   -- lr_slice:narrow(4, input:size(4) + self.pad_l + 1, self.pad_r):copy(input:narrow(4, 1, self.pad_r))

   -- zero out corners
   -- self.output:narrow(4, 1, self.pad_l):narrow(3, 1, self.pad_t):zero()
   -- self.output:narrow(4, 1, self.pad_l):narrow(3, input:size(3) + self.pad_t + 1, self.pad_b):zero()
   -- self.output:narrow(4, input:size(4) + self.pad_l + 1, self.pad_r):narrow(3, 1, self.pad_t):zero()
   -- self.output:narrow(4, input:size(4) + self.pad_l + 1, self.pad_r):narrow(3, input:size(3) + self.pad_t + 1, self.pad_b):zero()

   -----------------------------------------------------------------------
   -- About right, but fills corners with something .. 
   -----------------------------------------------------------------------

   self.output:narrow(3- ac,1,self.pad_t):copy(self.output:narrow(3- ac,input:size(3- ac) + 1,self.pad_t))
   self.output:narrow(3- ac,input:size(3- ac) + self.pad_t + 1,self.pad_b):copy(self.output:narrow(3- ac,self.pad_t + 1,self.pad_b))

   self.output:narrow(4- ac,1,self.pad_l):copy(self.output:narrow(4- ac,input:size(4- ac) + 1,self.pad_l))
   self.output:narrow(4- ac,input:size(4- ac) + self.pad_l + 1,self.pad_r):copy(self.output:narrow(4- ac,self.pad_l+1,self.pad_r))



   return self.output
end

-- function SpatialCircularPadding:updateGradInput(input, gradOutput)
--    if input:dim() ~= 4 and input:dim() ~= 3  then
--       error('input must be 3 or 4-dimensional')
--    end

--    -- Do it inplace to save memory
--    self.gradInput = nil
--    local cg_output = gradOutput

--    cg_output = cg_output:narrow(3, 1 + self.pad_t, cg_output:size(3) - self.pad_t)
--    cg_output = cg_output:narrow(3, 1, cg_output:size(3) - self.pad_b)
--    cg_output = cg_output:narrow(4, 1 + self.pad_l, cg_output:size(4) - self.pad_l)
--    cg_output = cg_output:narrow(4, 1, cg_output:size(4) - self.pad_r)


--    -- Border gradient
--    local tb_slice = gradOutput:narrow(4, self.pad_l+1,  input:size(4))
--    local lr_slice = gradOutput:narrow(3, self.pad_t+1,  input:size(3))

--    cg_output:narrow(3, input:size(3) - self.pad_t + 1, self.pad_t):add(tb_slice:narrow(3, 1, self.pad_t))
--    cg_output:narrow(3, 1, self.pad_b):add(tb_slice:narrow(3, input:size(3) + self.pad_t + 1, self.pad_b))

--    cg_output:narrow(4, input:size(4) - self.pad_l + 1, self.pad_l):add(lr_slice:narrow(4, 1, self.pad_l))
--    cg_output:narrow(4, 1, self.pad_r):add(lr_slice:narrow(4, input:size(4) + self.pad_l + 1, self.pad_r))


--    self.gradInput = cg_output
   
--    return self.gradInput
-- end


function SpatialCircularPadding:__tostring__()
  return torch.type(self) ..
      string.format('(l=%d,r=%d,t=%d,b=%d)', self.pad_l, self.pad_r,
                    self.pad_t, self.pad_b)
end