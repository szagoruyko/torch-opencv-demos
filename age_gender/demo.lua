local cv = require 'cv'
require 'cv.objdetect'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'loadcaffe'

assert(arg[1], 'please provide a path to haarcascade_frontalface_default.xml')
local cascade_path = arg[1]
local face_cascade = cv.CascadeClassifier{filename=cascade_path}

local fx = 0.5  -- rescale factor
local M = 227   -- input image size
local ages = {'0-2','4-6','8-13','15-20','25-32','38-43','48-53','60-'}

local download_list = {
  {name='age_net.caffemodel',     url='https://dl.dropboxusercontent.com/u/38822310/age_net.caffemodel'},
  {name='gender_net.caffemodel',  url='https://dl.dropboxusercontent.com/u/38822310/gender_net.caffemodel'},
  {name='deploy_age.prototxt',    url='https://git.io/vzZnX'},
  {name='deploy_gender.prototxt', url='https://git.io/vzZny'},
  {name='age_gender_mean.t7',     url='https://dl.dropboxusercontent.com/u/44617616/age_gender_mean.t7'}
}

for k,v in ipairs(download_list) do
  if not paths.filep(v.name) then os.execute('wget '..v.url..' -o '..v.name) end
end

local gender_net = loadcaffe.load('./deploy_gender.prototxt', './gender_net.caffemodel')
local age_net = loadcaffe.load('./deploy_age.prototxt', './age_net.caffemodel')

local img_mean = torch.load'./age_gender_mean.t7':permute(3,1,2):float()

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
