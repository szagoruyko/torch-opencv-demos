local cv = require 'cv'
require 'cv.objdetect'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'loadcaffe'

if not arg[1] then
    print[[
Usage: th demo.lua video_source [path-to-'haarcascade_frontalface_default.xml']

Where
  * video_source:

        Video source to capture.
        If "camera", then default camera is used.
        Otherwise, `video_source` is assumed to be a path to a video file.

  * path-to-'haarcascade_frontalface_default.xml':

        Optional argument, path to OpenCV's haarcascade_frontalface_default.xml.
        Use it if your `locate` command isn't able to find it it automatically.
]]
    os.exit(-1)
end

-- Viola-Jones face detector
local XMLTarget = 'haarcascades/haarcascade_frontalface_default.xml'
print('Looking for '..XMLTarget..'...')
local command = io.popen('locate '..XMLTarget, 'r')
local locateOutput = command:read()
local _, endIndex = locateOutput:find(XMLTarget)
local detectorParamsFile = locateOutput:sub(1, endIndex) or arg[2]
command:close()
assert(paths.filep(detectorParamsFile), 
       XMLTarget..' not found! Try using the second cmdline argument')

local face_cascade = cv.CascadeClassifier{detectorParamsFile}

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
  if not paths.filep(v.name) then os.execute('wget '..v.url..' -O '..v.name) end
end

local gender_net = loadcaffe.load('./deploy_gender.prototxt', './gender_net.caffemodel'):float()
local age_net = loadcaffe.load('./deploy_age.prototxt', './age_net.caffemodel'):float()

local img_mean = torch.load'./age_gender_mean.t7':permute(3,1,2):float()

local cap = cv.VideoCapture{arg[1] == 'camera' and 0 or arg[1]}
assert(cap:isOpened(), 'Failed to open '..arg[1])

local ok, frame = cap:read{}

if not ok then
  print("Couldn't retrieve frame!")
  os.exit(-1)
end

while true do
  local w = frame:size(2)
  local h = frame:size(1)

  local im2 = cv.resize{frame, fx=fx, fy=fx}
  cv.cvtColor{im2, dst=im2, code=cv.COLOR_BGR2GRAY}

  local faces = face_cascade:detectMultiScale{im2}
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

      cv.rectangle{frame, pt1={x, y+3}, pt2={x + w, y + h}, color={30,255,30}}
      cv.putText{
        frame,
        gender..': '..age,
        org={x, y},
        fontFace=cv.FONT_HERSHEY_DUPLEX,
        fontScale=1,
        color={255, 255, 0},
        thickness=1
      }
    end
  end

  cv.imshow{"torch-OpenCV Age&Gender demo", frame}
  ok = cap:read{frame}

  if cv.waitKey{1} >= 0 or not ok then break end
end
