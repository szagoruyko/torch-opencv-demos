local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.cudaobjdetect'
require 'cv.cudawarping' -- resize
require 'cv.cudaimgproc' -- cvtColor
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

local peopleNames = {}
for i = 1,N do
    peopleNames[i] = arg[2 + i] or 'Person #'..i
end

local stillLabeling = true
local pause = false
local currentFaceNumber

local function updateFaceNumber(faceNumber)
    if faceNumber > N then return end
    currentFaceNumber = faceNumber
    cv.setWindowTitle{
        'Stream window', 
        'Labeling '..peopleNames[faceNumber]..'\'s face. Press Enter when done'}
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
local frameCUDA = torch.CudaTensor(frame:size())
local scaleFactor = 0.5
local frameCUDAGray = torch.CudaTensor((#frame)[1] * scaleFactor, (#frame)[2] * scaleFactor)

-- Labeling loop
while stillLabeling do
    -- upload image to GPU and normalize it from [0..255] to [0..1]
    frameCUDA:copy(frame):div(255)
    -- convert to grayscale and store result in original image's blue channel
    cv.cuda.cvtColor{frameCUDA, frameCUDA:select(3,1), cv.COLOR_BGR2GRAY}
    -- resize it
    cv.cuda.resize{frameCUDA:select(3,1), dst=frameCUDAGray, fx=scaleFactor, fy=scaleFactor}
    
    --local faces = faceDetector:detectMultiScale{frameCUDAGray}
    local t = frameCUDAGray:float()

    cv.imshow{'Stream window', t}
   
    local key = cv.waitKey{20}

    if key >= 49 and key <= 57 then
        -- key is a digit: change current number of face to be labeled
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
