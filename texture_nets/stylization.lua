local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'

require 'nn'
require 'SpatialCircularPadding'
require 'NoiseFill'

require 'image'
require 'nn'
require 'xlua'

local opt = xlua.envparams{
   frame_height = 512,
   video_path = '',
   network = './starry.t7',
   type = 'float',
}


local cap = opt.video_path == '' and cv.VideoCapture{device=0} or cv.VideoCapture{filename=opt.video_path}
if not cap:isOpened() then
    print("Failed to open the default camera")
    os.exit(-1)
end

cv.namedWindow{opt.network, cv.WINDOW_AUTOSIZE}
local _,frame = cap:read{}

local function cast(x)
   if opt.type == 'float' then
      return x:float()
   elseif opt.type == 'cuda' then
      require 'cunn'
      return x:cuda()
   elseif opt.type == 'cl' then
      require 'clnn'
      return x:cl()
   end
end

local net = torch.load(opt.network)
cast(net):evaluate()

local function deprocess(img)
  local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
  img:add(mean_pixel:view(3, 1, 1):expandAs(img))
  return img
end


while true do
    local frame1 = frame:permute(3,1,2):float() / 255
    frame1 = image.scale(frame1, opt.frame_height)

    local input = frame1:view(1,table.unpack(frame1:size():totable()))
    
    local out = deprocess(net(cast(input))[1]:float())
    cv.imshow{opt.network, (torch.clamp(out:permute(2,3,1),0,255)):byte()}

    if cv.waitKey{30} >= 0 then break end
    cap:read{frame}
end
