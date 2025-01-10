#####################################################################
# Apellidos: Gómez Romero 
# Nombre: Alejandro José                                                  
######################## Instrucciones ##############################
#                                                                   #
# 1. RELLENA tus Apellidos y Nombre en la cabecera de este fichero. #
#                                                                   #
#                         ¡¡¡AHORA!!!                               #
#                                                                   #
# 2. CAMBIA el nombre del fichero a (en camel):                     #
#                                                                   #
#       Apellido1Apellido2Nombre.jl                                 #
#                                                                   #
# 3. NO BORRES nada del contenido actual. Añade cualquier función   #
#    adicional que uses: SOLO AQUELLAS QUE USES.                    #
#                                                                   #     
# 4. Se debe MANDAR ÚNICAMENTE el fichero .jl a: fsancho@us.es      #
#    y esperas confirmación de llegada.                             #
#                                                                   #
# 5. Hay funciones auxiliares (como onehot o escala) que no están   #
#    incluidas en las librerías.                                    #
#                                                                   #
# 6. COMENTA los códigos que escribas, no esperes que adivinación.  #
#                                                                   #
# 7. Ten en cuenta que yo evaluaré tu solución tal y como esté, lo  #
#    que no cargue no podrá ser evaluado con la nota máxima.        #
#                                                                   #
#####################################################################

using Pkg
Pkg.activate(".")
Pkg.add("StatsBase")

using CSV, DataFrames, StatsBase, Statistics, Random, Plots
using MLJ: unpack, partition

include("ID3.jl")
include("NN.jl"); using .NN;
include("Clust.jl")

#####################################################################
# Ej. 2 (2ptos)
#####################################################################

# Implementa el siguiente algoritmo: 

# Dado un dataset, D, de datos (x,y) donde x ∈ ℝ^n e y ∈ ℝ, el algoritmo 
# aproxima valores para nuevos vectores z ∈ ℝ^n de la siguiente forma:

# 1. Calcula los 2 puntos extremales de D para z, es decir, el más cercano a
#    z, x_m, y el más lejano a z, x_M.
#
# 2. Devuelve:
#            1  ⎛       1                     1            ⎞
#      y' = ——— ⎜ ———————————— ⋅ y_m +  ———————————— ⋅ y_M ⎟
#            2  ⎝ 1 + d(z,x_m)          1 + d(z,x_M)       ⎠
#
# donde d es la distancia euclídea, y (x_m, y_m), (x_M, y_M) ∈ D.
function alg2(D,z)


    
    # Calcula y devuelve la aproximación para z

    x_m,x_M = nothing,nothing
    y_m, y_M = nothing, nothing

    min_distance = Inf
    max_distance = -Inf

    for (x, y) in D
        distance = eucl(z, x)
        
        # Actualiza el punto más cercano
        if distance < min_distance
            min_distance = distance
            x_m, y_m = x, y
        end
        
        # Actualiza el punto más lejano
        if distance > max_distance
            max_distance = distance
            x_M, y_M = x, y
        end
    end
    
    f(x_m, x_M, y_m, y_M) = 1/2*(((1/(1+eucl(z, x_m))*y_m))+((1/(1+eucl(z, x_m))*y_M)))
    
    zs = [x_m; x_M; y_m; y_M]
    v =  [zs]
    yv = collect([f.(x_m, x_M, y_m, y_M)])

    red_i = NN.Network([6, 10, 1])

    red_f = NN.SGD(red_i, v, yv, 1000, 10, 0.4)
end




# Algunos Datasets de prueba:
D1X = [ round.(100 .* rand(2), digits = 1)  for _ in 1:100 ]
D1Y = round.(100 .* rand(100), digits = 1) 
D1  = collect(zip(D1X,D1Y))


alg2(D1, 20)


D2X = [ (10 .* rand(3)) for _ in 1:100 ]
D2Y = rand(100)
D2  = collect(zip(D2X,D2Y))

#####################################################################
# Ej. 3
#####################################################################

# El fichero `Guns.csv` tiene información acerca de ciertos valores 
# socio-económicos y judiciales de los estados de EEUU durante los años 
# 1977-1999. Concretamente: 
#
# `state` (estado), 
# `year` (año), 
# `violent` (tasa de delitos violentos), 
# `murder` (tasa de homicidios), 
# `robbery` (tasa de robos), 
# `prisoners` (tasa de encarcelamiento), 
# `afam` (% población afroamericana), 
# `cauc` (% población caucásica), 
# `male` (% población masculina), 
# `population` (población), 
# `income` (renta per cápita), 
# `density` (densidad), 
# `law` (si hay ley sobre armas en vigor ese año). 

# Usando los algoritmos que desees, de los que hemos visto en el curso, diseña 
# y ejecuta experimentos que proporcionen conocimiento adicional sobre el 
# dataset (puedes modificar el dataset como necesites).
function onehot(vals)
    labels = unique(vals)
    map(x -> onehot(x, labels), vals)
end

function escala(a1, b1, a2, b2)
    function escala(x)
        x1 = (x - a1) / (b1 - a1)
        x2 = x1 * (b2 - a2) + a2
        x2
    end
    return escala
end

function df_to_vectors(df)
    [collect(row) for row in eachrow(df)]
end

path = dirname(@__FILE__)
guns = CSV.read(path * "/Guns.csv", DataFrame, header=true)
onehot_law(law) = law == "yes" ? 1. : 0.

guns = select!(guns, Not([:rownames, :year]))

y,x = unpack(guns, ==(:law))
y = collect([onehot_law(i) for i in y ])
describe(y)

x = transform(x, :prisoners => ByRow(x -> x + 0.))
x = select!(x, Not([:prisoners, :state])) 

x
mapcols(c -> c .= escala(minimum(c), maximum(c), 0, 1).(c), x)

(Xtrain, Xtest), (Ytrain,Ytest) = partition((x,y),  0.8, rng=123, multi=true) 

Xtrain = df_to_vectors(Xtrain)
describe(Xtrain)
Xtrain

Xtest = df_to_vectors(Xtest)

Ytrain = df_to_vectors(Ytrain)

red_crabs = NN.Network([10, 4, 1])
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