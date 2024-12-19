using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Random
using CSV
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("MLJ")
using DataFrames
using MLJ: unpack, partition
include("NN.jl");
using .NN;
Pkg.add("Plots")
using Plots
plotly()  

function onehot(val, v)
    i      = findfirst(x -> x == val, v)
    res    = zeros(length(v))
    res[i] = 1
    res
end

function onehot(vals)
    labels = unique(vals)
    map(x -> onehot(x, labels), vals)
end

function df_to_vectors(df)
    [collect(row) for row in eachrow(df)]
end

function escala(a1, b1, a2, b2)
    function escala(x)
        x1 = (x - a1) / (b1 - a1)
        x2 = x1 * (b2 - a2) + a2
        x2
    end
    return escala
end

# Carga del dataset un dataframe.
crabs = CSV.read("./CSV/crabs.csv", DataFrame)
describe(crabs)

# Eliminamos columnas no utiles.
crabs = select!(crabs, Not([:rownames, :index]))
describe(crabs)

# Separación de Características/Objetivo
y, x = unpack(crabs, ==(:sex))

# Codificación Onehot del objetivo.
onehot_sex(sex) = sex == "M" ? 1. : 0.
onehot_sp(sp) = sp == "B" ? 1. : 0.

#crabs[!, :sp] = [onehot_sp(crabs[i, :sp]) for i in 1:nrow(crabs)]
#crabs[!, :sex] = [onehot_sex(crabs[i, :sex]) for i in 1:nrow(crabs)]

sexos = unique(y)
y = collect(onehot(y))
clases = unique(y)
# One hot Normaliz
# sexos = unique(y)
# y = collect(onehot(y))

# Columna: sp 2-categorica a 0/1
# x = transform(x, :sp => ByRow(x -> ((x=="B") ? 1. : 0.)) => :sp)
x[!, :sp] = [onehot_sp(x[i, :sp]) for i in 1:nrow(x)]

# Columnas númericas estandarizadas [1:0]
mapcols(c -> c .= escala(minimum(c), maximum(c), 0, 1).(c), x)

(Xtrain, Xtest), (Ytrain,Ytest) = partition((x,y),  0.8, rng=123, multi=true) 

Xtrain = df_to_vectors(Xtrain)
describe(Xtrain)
Xtrain

Xtest = df_to_vectors(Xtest)


# Entrenamiento
red_crabs = NN.Network([6, 4, 2])
NN.SGD(red_crabs, Xtrain, Ytrain, 1000, 10, 0.4)

function error_crabs(X, y, red)
    err2 = 0
    for (x1, y1) in zip(X, y)
        class_real = argmax(y1)
        y2         = NN.feed_forward(red, x1)
        class_red  = argmax(y2)
        marca      = ""
        if class_red != class_real
            err2  = err2 + 1
            marca = "*"
        end
        println("class = $class_real (r(x) = $class_red)$marca")
    end
    println("Error: $(err2/length(y))")
end

error_crabs(Xtrain, Ytrain, red_crabs)
error_crabs(Xtest, Ytest, red_crabs)

function calcula_MatrizConfusion(X_d, Y_d)
    M = zeros(length(clases), length(clases))
    for (x, y) in zip(X_d, Y_d)
        o_red  = NN.feed_forward(red_crabs, x)
        y_red  = argmax(o_red)
        y_real = argmax(y)
        if y_red != y_real
            M[y_real, y_red] = M[y_real, y_red] + 1
        end
    end
    p = heatmap(clases, clases, M, color=:greys, clim=(0, 5))
    display(p)
    return M
end

calcula_MatrizConfusion(Xtrain, Ytrain)
calcula_MatrizConfusion(Xtest, Ytest)

# EJERCICIO 2

iris = CSV.read("./CSV/iris.csv", DataFrame, header = true)
describe(iris)



y,x = unpack(iris, ==(:class))

y = collect(onehot(y))
clases = unique(y)


(Xtrain, Xtest), (Ytrain,Ytest) = partition((x,y),  0.8, rng=123, multi=true) 

Xtrain = df_to_vectors(Xtrain)
describe(Xtrain)

Xtest = df_to_vectors(Xtest)

n_iris = NN.Network([4,3,3])
NN.SGD(n_iris, Xtrain, Ytrain, 1000, 10, 0.4)

function error_crabs(X, y, red)
    err2 = 0
    for (x1, y1) in zip(X, y)
        class_real = argmax(y1)
        y2         = NN.feed_forward(red, x1)
        class_red  = argmax(y2)
        marca      = ""
        if class_red != class_real
            err2  = err2 + 1
            marca = "*"
        end
        println("class = $class_real (r(x) = $class_red)$marca")
    end
    println("Error: $(err2/length(y))")
end

error_crabs(Xtrain, Ytrain, n_iris)
error_crabs(Xtest, Ytest, n_iris)

function calcula_MatrizConfusion(X_d, Y_d, red)
    M = zeros(length(clases), length(clases))
    for (x, y) in zip(X_d, Y_d)
        o_red  = NN.feed_forward(red, x)
        y_red  = argmax(o_red)
        y_real = argmax(y)
        if y_red != y_real
            M[y_real, y_red] = M[y_real, y_red] + 1
        end
    end
    p = heatmap(clases, clases, M, color=:greys, clim=(0, 5))
    display(p)
    return M
end
calcula_MatrizConfusion(Xtrain, Ytrain, n_iris)
calcula_MatrizConfusion(Xtest, Ytest, n_iris)