local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'nn'
require 'loadcaffe'

local cap = cv.VideoCapture{device=0}
if not cap:isOpened() then
   print("Failed to open the default camera")
   os.exit(-1)
end

cv.namedWindow{winname="torch-OpenCV ImageNet classification demo", flags=cv.WINDOW_AUTOSIZE}
local _, frame = cap:read{}

-- Setting up network
proto_name = 'deploy.prototxt'
model_name = 'nin_imagenet.caffemodel'
img_mean_name = 'ilsvrc_2012_mean.t7'

prototxt_url = 'http://git.io/vIdRW'
model_url = 'https://www.dropbox.com/s/0cidxafrb2wuwxw/'..model_name
img_mean_url = 'https://www.dropbox.com/s/p33rheie3xjx6eu/'..img_mean_name

if not paths.filep(proto_name) then os.execute('wget '..prototxt_url..' -O '..proto_name) end
if not paths.filep(model_name) then os.execute('wget '..model_url)    end
if not paths.filep(img_mean_name) then os.execute('wget '..img_mean_url) end


print '==> Loading network'
-- Using network in network http://openreview.net/document/9b05a3bb-3a5e-49cb-91f7-0f482af65aea
local net = loadcaffe.load(proto_name, './nin_imagenet.caffemodel'):float()
local img_mean = torch.load'ilsvrc_2012_mean.t7'.img_mean:float()
local synset_words = torch.load('synset.t7','ascii')

local M = 227

while true do
   local w = frame:size(2)
   local h = frame:size(1)

   local crop = cv.getRectSubPix{image=frame, patchSize={h,h}, center={w/2, h/2}}
   local im = cv.resize{src=crop, dsize={256,256}}
   local im2 = im:float() - img_mean
   local I = cv.resize{src=im2, dsize={M,M}}:permute(3,1,2):clone()

   local _,classes = net:forward(I):view(-1):float():sort(true)

   for i=1,5 do
      cv.putText{
         img=crop,
         text = synset_words[classes[i]],
         org={10,10 + i * 25},
         fontFace=cv.FONT_HERSHEY_DUPLEX,
         fontScale=1,
         color={255, 255, 0},
         thickness=1
      }
   end

   cv.imshow{winname="torch-OpenCV ImageNet classification demo", image=crop}
   if cv.waitKey{30} >= 0 then break end

   cap:read{image=frame}
end
