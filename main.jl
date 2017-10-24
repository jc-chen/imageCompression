using(PyPlot)
include("imageKmeans.jl")

function quantizeImage(file,b)
    pixelsMtx = imread(file)
    k = 2^b
    (nRows,nCols,d) = size(pixelsMtx)
    W = zeros(k,d) #means
    model = kMeans(pixelsMtx,k,doPlot=true)
    y = model.predict(pixelsMtx)
    W = model.W
    return y,W,nRows,nCols
end

function deQuantizeImage(y,W,nRows,nCols)
    (k,d) = size(W)
    u, = size(y)
    Y = zeros(u,d)
    for i in 1:u
        Y[i,:] = W[y[i],:]
    end
    X = reshape(Y,nRows,nCols,d)
    imshow(X,cmap=nothing)
end

function main(file,clusters)
    (y,W,nRows,nCols) = quantizeImage(file,clusters)
    deQuantizeImage(y,W,nRows,nCols)
end

main(ARGS[1],ARGS[2])