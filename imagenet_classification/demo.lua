local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'nn'

local cap = cv.VideoCapture{device=0}
if not cap:isOpened() then
   print("Failed to open the default camera")
   os.exit(-1)
end

cv.namedWindow{winname="torch-OpenCV ImageNet classification demo", flags=cv.WINDOW_AUTOSIZE}
local _, frame = cap:read{}

print '==> Downloading image and network'
local image_url = 'http://upload.wikimedia.org/wikipedia/commons/e/e9/Goldfish3.jpg'
local network_url = 'https://www.dropbox.com/s/npmr5egvjbg7ovb/nin_nobn_final.t7'
local image_name = paths.basename(image_url)
local network_name = paths.basename(network_url)
if not paths.filep(image_name) then os.execute('wget '..image_url)   end
if not paths.filep(network_name) then os.execute('wget '..network_url)   end


print '==> Loading network'
-- Using network in network http://openreview.net/document/9b05a3bb-3a5e-49cb-91f7-0f482af65aea
local net = torch.load(network_name):unpack():float()
local synset_words = torch.load('synset.t7','ascii')

local M = 224

while true do
   local w = frame:size(2)
   local h = frame:size(1)

   local crop = cv.getRectSubPix{image=frame, patchSize={h,h}, center={w/2, h/2}}
   local im = cv.resize{src=crop, dsize={256,256}}:float():div(255)
   for i=1,3 do im:select(3,i):add(-net.transform.mean[i]):div(net.transform.std[i]) end
   local I = cv.resize{src=im, dsize={M,M}}:permute(3,1,2):clone()

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
