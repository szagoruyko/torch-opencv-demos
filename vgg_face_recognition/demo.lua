local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.cudaobjdetect'
require 'cv.cudaimgproc'
require 'nn'

if not arg[1] or not arg[2] then
    print[[
Usage: th demo.lua P N [Name1 Name2 ...]

Where
    P: Path to your `haarcascades_cuda/haarcascade_frontalface_default.xml`
    N: Number of different people to recognize (2..9)
    Name1, Name2, ...: Optional people names
]]
    os.exit(-1)
end

--local faceDetector = cv.cuda.CascadeClassifier{arg[1]}

print('Loading the network...')
local network = torch.load('./VGG_FACE.t7')
network:evaluate()

local capture = cv.VideoCapture{device=0}
assert(capture:isOpened(), 'Failed to open the default camera')

cv.namedWindow{'Stream window'}
cv.namedWindow{ 'Faces window'}
cv.setWindowTitle{'Faces window', 'Grabbed faces'}

-- **************************  Labeling the data **************************

local N = assert(tonumber(arg[2]))
local stillLabeling = true
local pause = false
local currentFaceNumber

local function updateFaceNumber(faceNumber)
    if faceNumber > N then return end
    currentFaceNumber = faceNumber
    cv.setWindowTitle{
        'Stream window', 
        'Labeling face #'..faceNumber..'. Press Enter when done'}
end

updateFaceNumber(1)

local faceSamples = {}
for i = 1,N do faceSamples[i] = {} end

local function enoughFaceSamples()
    local minNumber = 1e9
    for i = 1,N do minNumber = math.min(minNumber, #faceSamples[i]) end
    return minNumber >= 2
end

local _, frame = capture:read{}

-- Labeling loop
while stillLabeling do

   
    cv.imshow{'Stream window', frame}
   
    local key = cv.waitKey{20}

    if key >= 49 and key <= 57 then
        -- key is a digit: change face number to label
        updateFaceNumber(key-48)
    elseif key == 32 then
        -- key is Space  : set pause
        pause = not pause
    elseif key == 13 then
        -- key is Enter  : end labeling if there are at least 2 samples for each face
        if enoughFaceSamples() then stillLabeling = false end
    elseif key == 27 then
        -- key is Esc    : quit
        os.exit(0)
    end

    if not pause then capture:read{frame} end
end
