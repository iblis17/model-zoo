using Flux, MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated

x, y = traindata()
y = onehotbatch(y, 0:9)

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10, σ),
  softmax)

# using CuArrays
# x, y = cu(x), cu(y)
# m = mapleaves(cu, m)

using CLArrays.CLArray

cl2(xs) = NNlib.adapt(CLArray{Float32}, xs)

NNlib.adapt_(::Type{<:CLArray{T}}, xs::AbstractArray{<:Real}) where T<:AbstractFloat =
  isbits(xs) ? xs : convert(CLArray{T}, xs)

NNlib.adapt_(::Type{<:CLArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(CLArray, xs)

function NNlib.softmax(xs::CLArrays.CLArray)
  ys = exp.(xs)
  ys ./ sum(ys)
end

x, y = cl2(x), cl2(hcat(collect.(collect(y.data))...))
m = mapleaves(cl2, m)

loss(x, y) = Flux.mse(m(x), y)

accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

dataset = repeated((x, y), 200)
evalcb = () -> @show(loss(x, y))
opt = ADAM(params(m))

Flux.train!(loss, dataset, opt, cb = [evalcb])

accuracy(x, y)

# Test set accuracy
tx, ty = testdata()
ty = onehotbatch(ty, 0:9)
accuracy(tx, ty)
