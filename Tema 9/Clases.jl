using Pkg
Pkg.activate(".")
# Pkg.instantiate()
include("Clust.jl")
using CSV
using DataFrames
using Plots
plotly()

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

# Ejercicio de vinos.

# Primero cargamos el dataframe y eliminamos la columna clase 
#   1. Cargar DataFrame
#   2. Normalizamos los datos (escala)
vinos = CSV.read("./CSV/wine.csv", DataFrame, header = true)
X = select(vinos, Not(:Class))
#Convertir todas las comulnas del dataframe a float
X = mapcols(x -> convert(Vector{Float64}, x), X)

mapcols(c -> c.=escala(minimum(c), maximum(c), 0.0, 1.0).(c), X)
D  = [ collect(X[j, 1:end]) for j in 1:size(X, 1)]

Codo(D, 10,20)

Y = k_medias(D, 3)

function error_cluster(Cls, df)
    error = 0
    for k in 1:3 # Clusters
        for i in 1:length(Cls[k])
            for j in i+1:length(Cls[k])
                if df[Cls[k][i], :class] != df[Cls[k][j], :class]
                    error += 1
                end
            end
        end
    end
    # Calculamos el error como el número de pares que no coinciden de cluster / n^2, donde n es el número de elementos
    return error / (nrow(df)^2)
end