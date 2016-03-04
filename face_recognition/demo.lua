local cv = require 'cv'
require 'cv.highgui'       -- GUI: windows, mouse
require 'cv.videoio'       -- VideoCapture
require 'cv.imgproc'       -- rectangle, putText
require 'cv.cudaobjdetect' -- CascadeClassifier
require 'cv.cudawarping'   -- resize
require 'cv.cudaimgproc'   -- cvtColor
cv.ml = require 'cv.ml'    -- SVM

require 'cutorch'
require 'cunn'
require 'dpnn' -- nn.Inception

-------------------------------------------------------------------------------
-- Describe command line arguments
-------------------------------------------------------------------------------
if not arg[1] then
    print[[
Usage: th demo.lua [N [Name1 Name2 ...] ]

Where
    N: Number of different people to recognize (2..9)
    Name1, Name2, ...: Their names (optional)
]]
    os.exit(-1)
end

print('')
print('================== Use 1..9 keys to change the person to be labeled ==================')
print('==================        Double-click the face to label it         ==================')
print('==================         Press Space to pause the stream          ==================')
print('')

-------------------------------------------------------------------------------
-- Set up machine learning models
-------------------------------------------------------------------------------
-- Viola-Jones face detector
local XMLTarget = 'haarcascades_cuda/haarcascade_frontalface_default.xml'
print('Looking for '..XMLTarget..'...')
local command = io.popen('locate '..XMLTarget, 'r')
local locateOutput = command:read()
local _, endIndex = locateOutput:find(XMLTarget)
local detectorParamsFile = locateOutput:sub(1, endIndex)
command:close()
assert(paths.filep(detectorParamsFile), XMLTarget..' not found!')

local faceDetector = cv.cuda.CascadeClassifier{detectorParamsFile}

-- OpenFace convolutional neural network face descriptor
print('Loading the network...')
local networkURL = 'http://openface-models.storage.cmusatyalab.org/nn4.small2.v1.t7'
local networkName = paths.basename(networkURL)
if not paths.filep(networkName) then os.execute('wget '..networkURL) end

local network = torch.load(networkName)
network:cuda()   -- move network to GPU
local netInputSize = 96
local netOutputSize = 128
network:evaluate()
local netInput =    torch.CudaTensor(3, netInputSize, netInputSize)
local netInputHWC = torch.CudaTensor(netInputSize, netInputSize, 3)

-- SVM to classify descriptors in recognition phase
local svm = cv.ml.SVM{}
svm:setType         {cv.ml.SVM_C_SVC}
svm:setKernel       {cv.ml.SVM_POLY}
svm:setDegree       {2}
svm:setTermCriteria {{cv.TermCriteria_MAX_ITER, 100, 1e-6}}

-------------------------------------------------------------------------------
-- Set up video stream and GUI, unpack input arguments
-------------------------------------------------------------------------------
local capture = cv.VideoCapture{'/home/shrubb/Videos/FOB.mp4'}
assert(capture:isOpened(), 'Failed to open the default camera')

-- create two windows
cv.namedWindow{'Stream window'}
cv.namedWindow{ 'Faces window'}
cv.setWindowTitle{'Faces window', 'Grabbed faces'}
cv.moveWindow{'Stream window', 5, 5}
cv.moveWindow{'Faces window', 700, 100}

local N = assert(tonumber(arg[1]))
assert(N < 10 and N > 0 and N == math.floor(N))

-- prepare the "face gallery"
local thumbnailSize = 64
local maxThumbnails = 10
-- white background
local gallery = torch.ByteTensor(thumbnailSize*N, thumbnailSize*maxThumbnails + 100, 3):fill(255)
-- black stripes
for i = 1,N-1 do
    gallery:select(1, thumbnailSize*i):zero()
end
gallery:select(2, 100):zero()
gallery:select(2, 100 + 2*thumbnailSize):select(2, 1):fill(30)
gallery:select(2, 100 + 2*thumbnailSize):select(2, 2):fill(30)

local peopleNames = {}
for i = 1,N do
    peopleNames[i] = arg[1 + i] or 'Person #'..i
    cv.putText{
        gallery, peopleNames[i]:sub(1,10), {2, thumbnailSize*(i-1) + 36}, 
        fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color={160,30,30}}
end

local stillLabeling = true
local pause = false
local currentPerson

local function updateFaceNumber(faceNumber)
    if faceNumber > N then return end
    currentPerson = faceNumber
    cv.setWindowTitle{
        'Stream window', 
        'Labeling '..peopleNames[faceNumber]..'\'s face. Press Enter when done'}
end

updateFaceNumber(1)

local people = {}
for i = 1,N do people[i] = {} end

local function enoughFaces()
    local minNumber = 1e9
    for i = 1,N do minNumber = math.min(minNumber, #people[i]) end
    return minNumber >= 2
end

-- ByteTensor that comes from the camera
local _, frame = capture:read{}
-- CudaTensor that houses `frame`
local frameCUDA = torch.CudaTensor(frame:size())

local scaleFactor = 0.5
-- CudaTensor where the downsampled and desaturated `frameCUDA` resides
local frameCUDAGray = torch.CudaTensor((#frame)[1] * scaleFactor, (#frame)[2] * scaleFactor)

-- accomodation for cropped faces on GPU
local bigFaceCUDA = frameCUDA:clone()

-- RectArray describing the faces found
local faceRects

-------------------------------------------------------------------------------
-- Given a ByteTensor and bounding box, compute face descriptor with CNN on GPU
-------------------------------------------------------------------------------
local function getDescriptor(croppedFace, rect)
    -- copy face to GPU
    local smallFaceCUDA = bigFaceCUDA:narrow(1, 1, rect.height-2):narrow(2, 1, rect.width-2)
    smallFaceCUDA:copy(croppedFace)

    -- rescale it to network input size
    cv.cuda.resize{smallFaceCUDA, {netInputSize, netInputSize}, dst=netInputHWC}
    netInput:copy(netInputHWC:permute(3,1,2))

    -- pass it forward through CNN
    for i = 1,3 do
        netInput[i]:div(255)
    end
    return network:forward(netInput:view(1, 3, netInputSize, netInputSize)):float()
end

-------------------------------------------------------------------------------
-- The function executed at mouse double-click
-------------------------------------------------------------------------------
local function onMouse(event, x, y, flags)
    if not faceRects or event ~= cv.EVENT_LBUTTONDBLCLK or not stillLabeling then
        return
    end

    -- find a matching rectangle from faceRects
    for i = 1,faceRects.size do
        local f = faceRects.data[i]

        -- check if click location is inside bounding box
        if y >= f.y and y <= f.y + f.height and x >= f.x and x <= f.x + f.width then
            -- crop the face
            local croppedFace = 
                cv.getRectSubPix{
                    frame, 
                    {f.width-2, f.height-2}, 
                    {f.x + f.width/2, f.y + f.height/2}}

            if #people[currentPerson] < maxThumbnails then
                -- get slice for copying
                local faceInGallery = gallery
                    :narrow(1, 1 + thumbnailSize*(currentPerson-1), thumbnailSize)
                    :narrow(2, 101 + thumbnailSize * #people[currentPerson], thumbnailSize)
                -- show image in gallery
                cv.resize{croppedFace, {thumbnailSize, thumbnailSize}, dst=faceInGallery}
            end

            local descriptor = getDescriptor(croppedFace, f)

            -- save descriptor for future SVM training
            table.insert(people[currentPerson], descriptor)
            break
        end
    end
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
        faceRects = faceDetector:convert{faceRects}
        
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
        -----------------------------------------------------------------------
        -- Labeling phase
        -----------------------------------------------------------------------
        if key >= 49 and key <= 57 then
            -- key is a digit: change current number of face to be labeled
            updateFaceNumber(key-48)
        elseif key == 32 then
            -- key is Space  : set pause
            pause = not pause
        elseif key == 10 then
            -- key is Enter  : end labeling if there are at least 2 samples for each face
            if enoughFaces() then
                stillLabeling = false
                cv.setWindowTitle{'Stream window', 'Live Recognition'}

                -- prepare data for feeding to SVM
                local totalFaces = 0
                for i = 1,N do
                    totalFaces = totalFaces + #people[i]
                end

                local svmDataX = torch.FloatTensor(totalFaces, netOutputSize)
                local svmDataY = {}

                -- the data is traditionally presented row-wise
                for class = 1,N do
                    for i = 1,#people[class] do
                        table.insert(svmDataY, class)
                        svmDataX[#svmDataY]:copy(people[class][i])
                    end
                end
                svmDataY = torch.IntTensor(svmDataY)

                svm:train{svmDataX, cv.ml.ROW_SAMPLE, svmDataY}
            end
        elseif key == 27 then
            -- key is Esc    : quit
            os.exit(0)
        end
    else
        -----------------------------------------------------------------------
        -- Recognition phase
        -----------------------------------------------------------------------
        if key == 32 then
            -- key is Space  : set pause
            pause = not pause
        elseif key == 27 then
            -- key is Esc    : quit
            os.exit(0)
        end

        for i = 1,faceRects.size do
            local f = faceRects.data[i]

            -- crop the face
            local croppedFace = 
                cv.getRectSubPix{
                    frame, 
                    {f.width-2, f.height-2}, 
                    {f.x + f.width/2, f.y + f.height/2}}

            -- get descriptor
            local descriptor = getDescriptor(croppedFace, f)
            -- feed it to SVM, get class prediction
            local person = svm:predict{descriptor:view(1, netOutputSize)}
            -- draw predicted name above the rectangle
            cv.putText{
                frame, peopleNames[person], {f.x, f.y-3}, 
                fontFace=cv.FONT_HERSHEY_SIMPLEX, color={255,255,30}, 
                fontScale=1, thickness=2}
        end
    end

    cv.imshow{'Stream window', frame}
    cv.imshow{'Faces window', gallery}

    if not pause then capture:read{frame} end
end
