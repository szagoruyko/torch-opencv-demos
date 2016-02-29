local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.cudaobjdetect'
require 'cv.cudaimgproc'
require 'nn'

assert(arg[1], 'Please provide a path to haarcascades_cuda/haarcascade_frontalface_default.xml')
local faceDetector = cv.cuda.CascadeClassifier{arg[1]}

print('Loading the network...')
local network = torch.load('./VGG_FACE.t7')
network:evaluate()

os.exit(0)

local cap = cv.VideoCapture{device=0}
assert(cap:isOpened(), "Failed to open the default camera")

cv.namedWindow{winname="torch-OpenCV Age&Gender demo", flags=cv.WINDOW_AUTOSIZE}
local _, frame = cap:read{}

while true do
   local w = frame:size(2)
   local h = frame:size(1)

   local im2 = cv.resize{src=frame, fx=fx, fy=fx}
   cv.cvtColor{src=im2, dst=im2, code=cv.COLOR_BGR2GRAY}

   local faces = face_cascade:detectMultiScale{image = im2}
   for i=1,faces.size do
      local f = faces.data[i]
      local x = f.x/fx
      local y = f.y/fx
      local w = f.width/fx
      local h = f.height/fx

      -- crop and prepare image for convnets
      local crop = cv.getRectSubPix{
        image=frame,
        patchSize={w, h},
        center={x + w/2, y + h/2},
      }
      if crop then
      local im = cv.resize{src=crop, dsize={256,256}}:float()
      local im2 = im - img_mean
      local I = cv.resize{src=im2, dsize={M,M}}:permute(3,1,2):clone()

      -- classify
      local gender_out = gender_net:forward(I)
      local gender = gender_out[1] > gender_out[2] and 'M' or 'F'

      local age_out = age_net:forward(I)
      local _,id = age_out:max(1)
      local age = ages[id[1] ]

      cv.rectangle{img=frame, pt1={x, y}, pt2={x + w, y + h}, color={255,0,255,0}}
      cv.putText{
         img=frame,
         text = gender..': '..age,
         org={x, y},
         fontFace=cv.FONT_HERSHEY_DUPLEX,
         fontScale=1,
         color={255, 255, 0},
         thickness=1
      }
   end
   end

   cv.imshow{winname="torch-OpenCV Age&Gender demo", image=frame}
   if cv.waitKey{30} >= 0 then break end

   cap:read{image=frame}
end
