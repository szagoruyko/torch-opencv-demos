local cv = require 'cv'
require 'cv.highgui'       -- GUI: windows, mouse
require 'cv.videoio'       -- VideoCapture
require 'cv.cudaobjdetect' -- CascadeClassifier
require 'cv.cudawarping'   -- resize
require 'cv.cudaimgproc'   -- cvtColor
require 'cv.imgproc'       -- rectangle
cv.ml = require 'cv.ml'    -- SVM

require 'cutorch'
require 'nn'

-------------------------------------------------------------------------------
-- Describe command line arguments
-------------------------------------------------------------------------------
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

-------------------------------------------------------------------------------
-- Set up machine learning models
-------------------------------------------------------------------------------
-- Viola-Jones face detector
local faceDetector = cv.cuda.CascadeClassifier{arg[1]}

-- Convolutional neural network face descriptor by VGG
print('Loading the network...')
local network = torch.load('./VGG_FACE.t7')
network:evaluate()

-- SVM to classify descriptors in recognition phase
local svm = cv.ml.SVM{}
svm:setType         {cv.ml.SVM_C_SVC}
svm:setKernel       {cv.ml.SVM_LINEAR}
svm:setDegree       {2}
svm:setTermCriteria {{cv.TermCriteria_MAX_ITER, 100, 1e-6}}

-------------------------------------------------------------------------------
-- Set up video stream and GUI, unpack input arguments
-------------------------------------------------------------------------------
local capture = cv.VideoCapture{device=0}
assert(capture:isOpened(), 'Failed to open the default camera')

cv.namedWindow{'Stream window'}
cv.namedWindow{ 'Faces window'}
cv.setWindowTitle{'Faces window', 'Grabbed faces'}

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

local faces = {}
for i = 1,N do faces[i] = {} end

local function enoughFaces()
    local minNumber = 1e9
    for i = 1,N do minNumber = math.min(minNumber, #faces[i]) end
    return minNumber >= 2
end

local _, frame = capture:read{}
local frameCUDA = torch.CudaTensor(frame:size())

local scaleFactor = 0.5
local frameCUDAGray = torch.CudaTensor((#frame)[1] * scaleFactor, (#frame)[2] * scaleFactor)

local faceRects

local function onMouse(event, x, y, flags)
    if not faceRects or event ~= cv.EVENT_LBUTTONDBLCLK then
        return
    end

    print(x, y)
    -- find matching rectangle from faceRects
    -- crop it
    -- resize as needed
    -- pass it forward through CNN
    -- get a descriptor
    -- insert it into faces[currentFaceNumber]
end

cv.setMouseCallback{'Stream window', onMouse}

-------------------------------------------------------------------------------
-- The main loop
-------------------------------------------------------------------------------
while true do
    if not pause then
        -- upload image to GPU and normalize it from [0..255] to [0..1]
        frameCUDA:copy(frame):div(255)
        -- convert to grayscale and store result in original image's blue (first) channel
        cv.cuda.cvtColor{frameCUDA, frameCUDA:select(3,1), cv.COLOR_BGR2GRAY}
        -- resize it
        cv.cuda.resize{frameCUDA:select(3,1), dst=frameCUDAGray, fx=scaleFactor, fy=scaleFactor}
        
        -- detect faces in downsampled image
        faceRects = faceDetector:detectMultiScale{frameCUDAGray}
        -- convert faces to RectArray from OpenCV-CUDA's internal representation
        faceRects = faceDetector:convert{faces}
        
        -- draw faces
        for i = 1,faceRects.size do
            local f = faceRects.data[i]
            -- translate face coordinates to the original big image
            f.x      = f.x      / scaleFactor
            f.y      = f.y      / scaleFactor
            f.width  = f.width  / scaleFactor
            f.height = f.height / scaleFactor

            cv.rectangle{
                frame, {f.x, f.y}, {f.x + f.width, f.y + f.height}, 
                color = {30,30,180}, thickness = 2}
        end
    end

    local key = cv.waitKey{20}

    if stillLabeling then
        -- labeling phase
        if key >= 49 and key <= 57 then
            -- key is a digit: change current number of face to be labeled
            updateFaceNumber(key-48)
        elseif key == 32 then
            -- key is Space  : set pause
            pause = not pause
        elseif key == 13 then
            -- key is Enter  : end labeling if there are at least 2 samples for each face
            if enoughfaces() then stillLabeling = false end
        elseif key == 27 then
            -- key is Esc    : quit
            os.exit(0)
        end
    else
        -- recognition phase

        for i = 1,faceRects.size do
            local f = faceRects.data[i]

            -- crop the face
            -- resize as needed
            -- pass forward through CNN
            -- get descriptor
            -- feed it to SVM, get class prediction
            -- draw predicted name above the rectangle
        end
    end

    cv.imshow{'Stream window', frame}

    if not pause then capture:read{frame} end
end
